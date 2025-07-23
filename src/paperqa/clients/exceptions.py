from collections.abc import Callable

import aiohttp


class DOINotFoundError(Exception):
    def __init__(self, message="DOI not found") -> None:
        self.message = message
        super().__init__(self.message)


def make_flaky_ssl_error_predicate(host: str) -> Callable[[BaseException], bool]:
    def predicate(exc: BaseException) -> bool:
        # Seen with both Semantic Scholar and Crossref:
        # > aiohttp.client_exceptions.ClientConnectorError:
        # > Cannot connect to host api.host.org:443 ssl:default
        # > [nodename nor servname provided, or not known]
        # SEE: https://github.com/aio-libs/aiohttp/blob/v3.10.5/aiohttp/client_exceptions.py#L193-L196
        return isinstance(exc, aiohttp.ClientConnectorError) and exc.host == host

    return predicate
