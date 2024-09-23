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
    RateLimitItemPerHour,
    RateLimitItemPerMinute,
    RateLimitItemPerSecond,
)
from limits.aio.storage import MemoryStorage, RedisStorage
from limits.aio.strategies import MovingWindowRateLimiter

logger = logging.getLogger(__name__)

GLOBAL_RATE_LIMITER_TIMEOUT = float(os.environ.get("RATE_LIMITER_TIMOUT", "30"))
# RATE_CONFIG keys are tuples, corresponding to a namespace and primary key.
# Anything defined with MATCH_ALL variable, will match all non-matched requests for that namespace.
# For the "get" namespace, all primary key urls will be parsed down to the domain level.
# For example, you're trying to do a get request to "https://google.com", "google.com" will get
# its own limit, and it will use the ("get", MATCH_ALL) for its limits.
# machine_id is a unique identifier for the machine making the request, it's used to limit the
# rate of requests per machine. If the machine_id is not in the NO_PROXY_EXTENSIONS list, then
# the dynamic IP of the machine will be used to limit the rate of requests, otherwise the
# user input proxy_id will be used.
# NOTICE: When liteLLM is implemented, LLM-client limits can be set via the RouterAPI, for now we only
# have a custom client limit for openAI based embedding models
MATCH_ALL = None
MATCH_ALL_INPUTS = Literal[None]
MATCH_MACHINE_ID = "<machine_id>"

FALLBACK_RATE_LIMIT = RateLimitItemPerSecond(3, 1)
TOKEN_FALLBACK_RATE_LIMIT = RateLimitItemPerMinute(30_000, 1)

RATE_CONFIG: dict[tuple[str, str | MATCH_ALL_INPUTS], RateLimitItem] = {
    ("get", "api.crossref.org"): RateLimitItemPerSecond(30, 1),
    ("get", "api.semanticscholar.org"): RateLimitItemPerSecond(15, 1),
    ("client", MATCH_ALL): TOKEN_FALLBACK_RATE_LIMIT,
    # MATCH_MACHINE_ID is a placeholder for the machine_id passed in by the caller
    (f"get|{MATCH_MACHINE_ID}", MATCH_ALL): FALLBACK_RATE_LIMIT,
}

UNKNOWN_IP: str = "0.0.0.0"  # noqa: S104


class GlobalRateLimiter:

    WAIT_INCREMENT: ClassVar[float] = 0.01  # seconds
    IP_CHECK_SERVICES: ClassVar[Collection[str]] = {
        "https://api.ipify.org",
        "https://ifconfig.me",
        "http://icanhazip.com",
        "https://ipecho.net/plain",
    }
    # top sources pulled from prod log proxy failures
    NO_PROXY_EXTENSIONS: ClassVar[Collection[str]] = {
        ".gov",
        ".uk",
        "doi.org",
        "cyberleninka.org",
        ".de",
        ".jp",
        ".ro",
        "microsoft.com",
        "cambridge.org",
    }

    def __init__(
        self,
        rate_config: dict[
            tuple[str, str | MATCH_ALL_INPUTS], RateLimitItem
        ] = RATE_CONFIG,
        use_in_memory: bool = False,
    ):
        self.rate_config = rate_config
        self.use_in_memory = use_in_memory
        self._storage = None
        self._rate_limiter = None
        self._current_ip: str | None = None

    @staticmethod
    async def get_outbound_ip(session: aiohttp.ClientSession, url: str) -> str | None:
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(5)) as response:
                if response.status == 200:
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
                    else:
                        logger.error("Failed to retrieve IP from all services")
                        self._current_ip = UNKNOWN_IP
        return self._current_ip  # type: ignore[return-value]

    @property
    def storage(self):
        if self._storage is None:
            if os.environ.get("REDIS_URL") and not self.use_in_memory:
                self._storage = RedisStorage(f"async+redis://{os.environ['REDIS_URL']}")
                logger.info("Connected to redis instance for rate limiting.")
            else:
                self._storage = MemoryStorage()
                logger.info("Using in-memory rate limiter.")

        return self._storage

    @property
    def rate_limiter(self):
        if self._rate_limiter is None:
            self._rate_limiter = MovingWindowRateLimiter(self.storage)
        return self._rate_limiter

    async def parse_namespace_and_primary_key(
        self, namespace_and_key: tuple[str, str | MATCH_ALL_INPUTS], proxy_id: int = 0
    ) -> tuple[str, str | MATCH_ALL_INPUTS]:
        """Turn get key into a namespace and primary-key."""
        namespace, primary_key = namespace_and_key

        if namespace.startswith("get") and primary_key is not None:
            # for URLs to be parsed correctly, they need a protocol
            if not primary_key.startswith(("http://", "https://")):
                primary_key = "https://" + primary_key

            primary_key = urlparse(primary_key).netloc or urlparse(primary_key).path

            if any(ext in primary_key for ext in self.NO_PROXY_EXTENSIONS):
                namespace = f"{namespace}|{await self.outbount_ip()}"
            else:
                namespace = f"{namespace}|{proxy_id}"

        return namespace, primary_key

    def parse_rate_limits_and_namespace(
        self,
        namespace: str,
        primary_key: str | MATCH_ALL_INPUTS,
    ) -> tuple[RateLimitItem, str]:
        """Get rate limit and new namespace for a given namespace and primary_key."""
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
    ) -> tuple[RateLimitItem, tuple[str, str | MATCH_ALL_INPUTS]]:
        """Parse the rate limit item from a redis/in-memory key.

        Note the key is created with RateLimitItem.key_for(*identifiers),
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
        self,
    ) -> list[tuple[RateLimitItem, tuple[str, str | MATCH_ALL_INPUTS]]]:
        """Returns a list of current RateLimitItems with tuples of namespace and primary key"""
        host, port = os.environ.get("REDIS_URL", ":").split(":", maxsplit=2)

        if not (host and port):
            raise ValueError(f'Invalid REDIS_URL: {os.environ.get("REDIS_URL")}.')

        client = Redis(host=host, port=port)

        cursor = b"0"
        matching_keys = []

        while cursor:
            cursor, keys = await client.scan(
                cursor, match=f"{self.storage.PREFIX}*", count=100
            )
            matching_keys.extend(keys)

        await client.quit()

        return [self.parse_key(key.decode("utf-8")) for key in matching_keys]

    def get_in_memory_limit_keys(
        self,
    ) -> list[tuple[RateLimitItem, tuple[str, str | MATCH_ALL_INPUTS]]]:
        """Returns a list of current RateLimitItems with tuples of namespace and primary key"""
        return [self.parse_key(key) for key in self.storage.events]

    async def get_limit_keys(
        self,
    ) -> list[tuple[RateLimitItem, tuple[str, str | MATCH_ALL_INPUTS]]]:
        if os.environ.get("REDIS_URL") and not self.use_in_memory:
            return await self.get_rate_limit_keys()
        return self.get_in_memory_limit_keys()

    async def rate_limit_status(self):

        limit_status = {}

        for rate_limit, (namespace, primary_key) in await self.get_limit_keys():
            period_start, n_items_in_period = await self.storage.get_moving_window(
                rate_limit.key_for(*(namespace, primary_key)),
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
        namespace_and_key: tuple[str, str | MATCH_ALL_INPUTS],
        rate_limit: RateLimitItem | None = None,
        proxy_id: int = 0,
        timeout: float = GLOBAL_RATE_LIMITER_TIMEOUT,
        weight: int = 1,
    ) -> None:
        """Returns when the limit is satisfied for the namespace_and_key.
        
        Args:
            `namespace_and_key` is composed of a tuple with namespace (e.g. "get") and a primary-key (e.g. "arxiv.org").
            namespaces can be nested with multiple '|', primary-keys in the "get" namespace will be stripped to the domain.

            `proxy_id` will be used to modify the namespace of get requests if
            the primary key is not in the NO_PROXY_EXTENSIONS list.
            Otherwise, the outbound IP will be used to modify the namespace.

            The proxy_id / outbound IP are referred to as the machine_id.

            `rate_limit` is the rate limit to be used for the namespace and primary-key.
            
            `timeout` is the maximum time to wait for the rate limit to be satisfied.

            `weight` is the cost of the request, default is 1. (could be tokens for example)

        returns if the limit is satisfied or times out via a TimeoutError.

        """
        namespace, primary_key = await self.parse_namespace_and_primary_key(
            namespace_and_key, proxy_id=proxy_id
        )
        
        _rate_limit, new_namespace = self.parse_rate_limits_and_namespace(
            namespace, primary_key
        )

        rate_limit = rate_limit or _rate_limit

        while True:
            elapsed = 0.0
            while (
                not (
                    await self.rate_limiter.test(rate_limit, new_namespace, primary_key, cost=weight)
                )
                and elapsed < timeout
            ):
                await asyncio.sleep(self.WAIT_INCREMENT)
                elapsed += self.WAIT_INCREMENT

            if elapsed >= timeout:
                raise TimeoutError(
                    f"Timeout ({elapsed} secs): rate limit for key: {namespace_and_key}"
                )

            # If the rate limit hit is False, then we're violating the limit, so we
            # need to wait again. This can happen in race conditions.
            if await self.rate_limiter.hit(rate_limit, new_namespace, primary_key, cost=weight):
                break
            timeout = max(timeout - elapsed, 1.0)


GLOBAL_LIMITER = GlobalRateLimiter()
