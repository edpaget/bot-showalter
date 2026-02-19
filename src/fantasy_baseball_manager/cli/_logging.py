import logging
import sys

_THIRD_PARTY_LOGGERS = ("httpx", "httpcore", "urllib3", "filelock")


def configure_logging(*, verbose: bool = False) -> None:
    """Configure root logger for CLI output on stderr."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s — %(message)s", datefmt="%H:%M:%S"))
    root.addHandler(handler)

    for name in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.NOTSET if verbose else logging.WARNING)
