import logging
import sys

from fantasy_baseball_manager.cli._logging import configure_logging


class TestConfigureLogging:
    def setup_method(self) -> None:
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)

    def test_default_sets_info_level(self) -> None:
        configure_logging()
        assert logging.getLogger().level == logging.INFO

    def test_verbose_sets_debug_level(self) -> None:
        configure_logging(verbose=True)
        assert logging.getLogger().level == logging.DEBUG

    def test_third_party_suppressed_to_warning(self) -> None:
        configure_logging()
        for name in ("httpx", "httpcore", "pybaseball", "urllib3", "filelock"):
            assert logging.getLogger(name).level == logging.WARNING

    def test_third_party_not_suppressed_when_verbose(self) -> None:
        configure_logging(verbose=True)
        for name in ("httpx", "httpcore", "pybaseball", "urllib3", "filelock"):
            assert logging.getLogger(name).level == logging.NOTSET

    def test_handler_writes_to_stderr(self) -> None:
        configure_logging()
        root = logging.getLogger()
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.StreamHandler)
        assert handler.stream is sys.stderr

    def test_idempotent(self) -> None:
        configure_logging()
        configure_logging()
        assert len(logging.getLogger().handlers) == 1
