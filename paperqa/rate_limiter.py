import asyncio
import logging
import os
from collections.abc import Collection
from typing import ClassVar, Literal
from urllib.parse import urlparse

import aiohttp
from coredis import Redis
from limits import (
    RateLimitItem,
    RateLimitItemPerMinute,
    RateLimitItemPerSecond,
)
from limits import (
    parse as limit_parse,
)
from limits.aio.storage import MemoryStorage, RedisStorage
from limits.aio.strategies import MovingWindowRateLimiter

from .clients.crossref import CROSSREF_BASE_URL
from .clients.semantic_scholar import SEMANTIC_SCHOLAR_BASE_URL

logger = logging.getLogger(__name__)

GLOBAL_RATE_LIMITER_TIMEOUT = float(os.environ.get("RATE_LIMITER_TIMEOUT", "60"))

MATCH_ALL = None
MatchAllInputs = Literal[None]
MATCH_MACHINE_ID = "<machine_id>"

FALLBACK_RATE_LIMIT = RateLimitItemPerSecond(3, 1)
TOKEN_FALLBACK_RATE_LIMIT = RateLimitItemPerMinute(30_000, 1)

# RATE_CONFIG keys are tuples, corresponding to a namespace and primary key.
# Anything defined with MATCH_ALL variable, will match all non-matched requests for that namespace.
# For the "get" namespace, all primary key urls will be parsed down to the domain level.
# For example, you're trying to do a get request to "https://google.com", "google.com" will get
# its own limit, and it will use the ("get", MATCH_ALL) for its limits.
# machine_id is a unique identifier for the machine making the request, it's used to limit the
# rate of requests per machine. If the primary_key is in the NO_MACHINE_ID_EXTENSIONS list, then
# the dynamic IP of the machine will be used to limit the rate of requests, otherwise the
# user input machine_id will be used.

RATE_CONFIG: dict[tuple[str, str | MatchAllInputs], RateLimitItem] = {
    ("get", CROSSREF_BASE_URL): RateLimitItemPerSecond(30, 1),
    ("get", SEMANTIC_SCHOLAR_BASE_URL): RateLimitItemPerSecond(15, 1),
    ("client", MATCH_ALL): TOKEN_FALLBACK_RATE_LIMIT,
    # MATCH_MACHINE_ID is a sentinel for the machine_id passed in by the caller
    (f"get|{MATCH_MACHINE_ID}", MATCH_ALL): FALLBACK_RATE_LIMIT,
}

UNKNOWN_IP: str = "0.0.0.0"  # noqa: S104


class GlobalRateLimiter:
    """Rate limiter for all requests within or between processes.

    Supports both Redis and in-memory storage.
    'Global' refers to being able to limit the rate
    of requests across processes with Redis.
    """

    WAIT_INCREMENT: ClassVar[float] = 0.01  # seconds
    # list of public free outbount IP services
    # generated initially w. claude, then filtered
    IP_CHECK_SERVICES: ClassVar[Collection[str]] = {
        "https://api.ipify.org",
        "https://ifconfig.me",
        "http://icanhazip.com",
        "https://ipecho.net/plain",
    }
    # the following will use IP scope for limiting, rather
    # than user input machine ID
    NO_MACHINE_ID_EXTENSIONS: ClassVar[Collection[str]] = {"crossref.org"}

    def __init__(
        self,
        rate_config: (
            None | dict[tuple[str, str | MatchAllInputs], RateLimitItem]
        ) = None,
        use_in_memory: bool = False,
    ):
        self.rate_config = RATE_CONFIG if rate_config is None else rate_config
        self.use_in_memory = use_in_memory
        self._storage: RedisStorage | MemoryStorage | None = None
        self._rate_limiter: MovingWindowRateLimiter | None = None
        self._current_ip: str | None = None

    @staticmethod
    async def get_outbound_ip(session: aiohttp.ClientSession, url: str) -> str | None:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(5)) as response:
                if response.ok:
                    return await response.text()
        except TimeoutError:
            logger.warning(f"Timeout occurred while connecting to {url}")
        except aiohttp.ClientError:
            logger.warning(f"Error occurred while connecting to {url}.", exc_info=True)
        return None

    async def outbount_ip(self) -> str:
        if self._current_ip is None:
            async with aiohttp.ClientSession() as session:
                for service in self.IP_CHECK_SERVICES:
                    ip = await self.get_outbound_ip(session, service)
                    if ip:
                        logger.info(f"Successfully retrieved IP from {service}")
                        self._current_ip = ip.strip()
                        break
                if self._current_ip is None:
                    logger.error("Failed to retrieve IP from all services")
                    self._current_ip = UNKNOWN_IP
        return self._current_ip

    @property
    def storage(self) -> RedisStorage | MemoryStorage:
        if self._storage is None:
            if os.environ.get("REDIS_URL") and not self.use_in_memory:
                self._storage = RedisStorage(f"async+redis://{os.environ['REDIS_URL']}")
                logger.info("Connected to redis instance for rate limiting.")
            else:
                self._storage = MemoryStorage()
                logger.info("Using in-memory rate limiter.")

        return self._storage

    @property
    def rate_limiter(self) -> MovingWindowRateLimiter:
        if self._rate_limiter is None:
            self._rate_limiter = MovingWindowRateLimiter(self.storage)
        return self._rate_limiter

    async def parse_namespace_and_primary_key(
        self, namespace_and_key: tuple[str, str], machine_id: int = 0
    ) -> tuple[str, str]:
        """Turn namespace_and_key tuple into a namespace and primary-key.

        If using a namespace starting with "get", then the primary key will be url parsed.
        "get" namespaces will also have their machine_ids appended to the namespace here,
        unless the primary key is in the NO_MACHINE_ID_EXTENSIONS list, in which case
        the outbound IP will be used.
        """
        namespace, primary_key = namespace_and_key

        if namespace.startswith("get") and primary_key is not None:
            # for URLs to be parsed correctly, they need a protocol
            if not primary_key.startswith(("http://", "https://")):
                primary_key = "https://" + primary_key

            primary_key = urlparse(primary_key).netloc or urlparse(primary_key).path

            if any(ext in primary_key for ext in self.NO_MACHINE_ID_EXTENSIONS):
                namespace = f"{namespace}|{await self.outbount_ip()}"
            else:
                namespace = f"{namespace}|{machine_id}"

        return namespace, primary_key

    def parse_rate_limits_and_namespace(
        self,
        namespace: str,
        primary_key: str | MatchAllInputs,
    ) -> tuple[RateLimitItem, str]:
        """Get rate limit and new namespace for a given namespace and primary_key.

        This parsing logic finds the correct rate limits for a namespace/primary_key.
        It's a bit complex due to the <MATCH ALL> and <MATCH MACHINE ID> placeholders.
        These allow users to match

        """
        # the namespace may have a machine_id in it -- we replace if that's the case
        namespace_w_stub_machine_id = namespace
        namespace_w_machine_id_stripped = namespace

        # strip off the machine_id, and replace it with the MATCH_MACHINE_ID placeholder
        if namespace.startswith("get"):
            machine_id = namespace.split("|")[-1]
            if machine_id != "get":
                namespace_w_stub_machine_id = namespace.replace(
                    machine_id, MATCH_MACHINE_ID, 1
                )
                # try stripping the machine id for the namespace for shared limits
                # i.e. matching to one rate limit across ALL machines
                # these limits are in RATE_CONFIG WITHOUT a MATCH_MACHINE_ID placeholder
                namespace_w_machine_id_stripped = "|".join(namespace.split("|")[:-1])

        # here we want to use namespace_w_machine_id_stripped -- the rate should be shared
        # this needs to be checked first, since it's more specific than the stub machine id
        if (namespace_w_machine_id_stripped, primary_key) in self.rate_config:
            return (
                self.rate_config[(namespace_w_machine_id_stripped, primary_key)],
                namespace_w_machine_id_stripped,
            )
        # we keep the old namespace if we match on the namespace_w_stub_machine_id
        if (namespace_w_stub_machine_id, primary_key) in self.rate_config:
            return (
                self.rate_config[(namespace_w_stub_machine_id, primary_key)],
                namespace,
            )
        # again we only want the original namespace, keep the old namespace
        if (namespace_w_stub_machine_id, MATCH_ALL) in self.rate_config:
            return (
                self.rate_config[(namespace_w_stub_machine_id, MATCH_ALL)],
                namespace,
            )
        # again we want to use the stripped namespace if it matches
        if (namespace_w_machine_id_stripped, MATCH_ALL) in self.rate_config:
            return (
                self.rate_config[(namespace_w_machine_id_stripped, MATCH_ALL)],
                namespace_w_machine_id_stripped,
            )
        return FALLBACK_RATE_LIMIT, namespace

    def parse_key(
        self, key: str
    ) -> tuple[RateLimitItem, tuple[str, str | MatchAllInputs]]:
        """Parse the rate limit item from a redis/in-memory key.

        Args:
            key (str): is created with RateLimitItem.key_for(*identifiers),
            the first key is the namespace, then the next two will be our identifiers.

        """
        namespace, primary_key = key.split("/")[1:3]
        rate_limit, new_namespace = self.parse_rate_limits_and_namespace(
            namespace, primary_key
        )
        return (
            rate_limit,
            (new_namespace, primary_key),
        )

    async def get_rate_limit_keys(
        self, cursor_scan_count: int = 100
    ) -> list[tuple[RateLimitItem, tuple[str, str | MatchAllInputs]]]:
        """Returns a list of current RateLimitItems with tuples of namespace and primary key."""
        host, port = os.environ.get("REDIS_URL", ":").split(":", maxsplit=2)

        if not (host and port):
            raise ValueError(f'Invalid REDIS_URL: {os.environ.get("REDIS_URL")}.')

        if not isinstance(self.storage, RedisStorage):
            raise NotImplementedError(
                "get_rate_limit_keys only works with RedisStorage."
            )

        client = Redis(host=host, port=int(port))

        try:
            cursor: int | bytes = b"0"
            matching_keys: list[bytes] = []
            while cursor:
                cursor, keys = await client.scan(
                    int(cursor),
                    match=f"{self.storage.PREFIX}*",
                    count=cursor_scan_count,
                )
                matching_keys.extend(list(keys))
        finally:
            await client.quit()

        return [self.parse_key(key.decode()) for key in matching_keys]

    def get_in_memory_limit_keys(
        self,
    ) -> list[tuple[RateLimitItem, tuple[str, str | MatchAllInputs]]]:
        """Returns a list of current RateLimitItems with tuples of namespace and primary key."""
        if not isinstance(self.storage, MemoryStorage):
            raise NotImplementedError(
                "get_in_memory_limit_keys only works with MemoryStorage."
            )
        return [self.parse_key(key) for key in self.storage.events]

    async def get_limit_keys(
        self,
    ) -> list[tuple[RateLimitItem, tuple[str, str | MatchAllInputs]]]:
        if os.environ.get("REDIS_URL") and not self.use_in_memory:
            return await self.get_rate_limit_keys()
        return self.get_in_memory_limit_keys()

    async def rate_limit_status(self):

        limit_status = {}

        for rate_limit, (namespace, primary_key) in await self.get_limit_keys():
            period_start, n_items_in_period = await self.storage.get_moving_window(
                rate_limit.key_for(*(namespace, primary_key or "")),
                rate_limit.amount,
                rate_limit.get_expiry(),
            )
            limit_status[(namespace, primary_key)] = {
                "period_start": period_start,
                "n_items_in_period": n_items_in_period,
                "period_seconds": rate_limit.GRANULARITY.seconds,
                "period_name": rate_limit.GRANULARITY.name,
                "period_cap": rate_limit.amount,
            }
        return limit_status

    async def try_acquire(
        self,
        namespace_and_key: tuple[str, str],
        rate_limit: RateLimitItem | str | None = None,
        machine_id: int = 0,
        acquire_timeout: float = GLOBAL_RATE_LIMITER_TIMEOUT,
        weight: int = 1,
        raise_impossible_limits: bool = False,
    ) -> None:
        """Returns when the limit is satisfied for the namespace_and_key.

        Args:
            namespace_and_key (:obj:`tuple[str, str]`): is
                composed of a tuple with namespace (e.g. "get") and a primary-key
                (e.g. "arxiv.org"). namespaces can be nested with multiple '|',
                primary-keys in the "get" namespace will be stripped to the domain.
            rate_limit (:obj:`RateLimitItem | str | None`, optional): Optional
                RateLimitItem to be used for the namespace and primary-key.
                If not provided, RATE_CONFIG will be used to find the rate limit.
                Can also use a string of the form:
                [count] [per|/] [n (optional)] [second|minute|hour|day|month|year]
            machine_id (:obj:`int`, optional): will be used to modify the namespace
                of GET requests if the primary key is not in the
                NO_MACHINE_ID_EXTENSIONS list. In that case, the outbound IP will be
                used to modify the namespace.
            acquire_timeout (:obj:`float`, optional): is the maximum time (in seconds) to
                wait for the rate limit to be satisfied.
            weight (:obj:`int`, optional): is the cost of the request,
                default is 1. (could be tokens for example)
            raise_impossible_limits (:obj:`bool`, optional): flag will raise a
                ValueError for weights that exceed the rate.

        Raises:
            TimeoutError: if the acquire_timeout is exceeded.
            ValueError: if the weight exceeds the rate limit and raise_impossible_limits is True.
        """
        namespace, primary_key = await self.parse_namespace_and_primary_key(
            namespace_and_key, machine_id=machine_id
        )

        _rate_limit, new_namespace = self.parse_rate_limits_and_namespace(
            namespace, primary_key
        )

        if isinstance(rate_limit, str):
            rate_limit = limit_parse(rate_limit)

        rate_limit = rate_limit or _rate_limit

        if rate_limit.amount < weight and raise_impossible_limits:
            raise ValueError(
                f"Weight ({weight}) > RateLimit ({rate_limit}), cannot satisfy rate"
                " limit."
            )
        while True:
            elapsed = 0.0
            while (
                not (
                    await self.rate_limiter.test(
                        rate_limit,
                        new_namespace,
                        primary_key,
                        cost=min(weight, rate_limit.amount),
                    )
                )
                and elapsed < acquire_timeout
            ):
                await asyncio.sleep(self.WAIT_INCREMENT)
                elapsed += self.WAIT_INCREMENT
            if elapsed >= acquire_timeout:
                raise TimeoutError(
                    f"Timeout ({elapsed} secs): rate limit for key: {namespace_and_key}"
                )

            # If the rate limit hit is False, then we're violating the limit, so we
            # need to wait again. This can happen in race conditions.
            if await self.rate_limiter.hit(
                rate_limit,
                new_namespace,
                primary_key,
                cost=min(weight, rate_limit.amount),
            ):
                # we need to keep trying when we have an "impossible" limit
                if rate_limit.amount < weight:
                    weight -= rate_limit.amount
                    acquire_timeout = max(acquire_timeout - elapsed, 1.0)
                    continue
                break
            acquire_timeout = max(acquire_timeout - elapsed, 1.0)


GLOBAL_LIMITER = GlobalRateLimiter()
