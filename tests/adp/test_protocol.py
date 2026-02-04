"""Tests for ADPSource protocol."""

from datetime import UTC, datetime

from fantasy_baseball_manager.adp.models import ADPData, ADPEntry
from fantasy_baseball_manager.adp.protocol import ADPSource


class TestADPSourceProtocol:
    """Tests for ADPSource protocol."""

    def test_protocol_defines_fetch_adp(self) -> None:
        """Test that ADPSource protocol requires fetch_adp() -> ADPData."""

        class FakeADPSource:
            """Fake implementation to test protocol."""

            def fetch_adp(self) -> ADPData:
                return ADPData(
                    entries=(ADPEntry(name="Test", adp=1.0, positions=("OF",)),),
                    fetched_at=datetime.now(UTC),
                )

        source: ADPSource = FakeADPSource()
        result = source.fetch_adp()

        assert isinstance(result, ADPData)
        assert len(result.entries) == 1

    def test_protocol_is_structural(self) -> None:
        """Test that any class with fetch_adp() satisfies the protocol."""

        class AnotherSource:
            """Another implementation without explicit inheritance."""

            def fetch_adp(self) -> ADPData:
                return ADPData(
                    entries=(),
                    fetched_at=datetime.now(UTC),
                    source="other",
                )

        def accepts_adp_source(source: ADPSource) -> str:
            return source.fetch_adp().source

        source = AnotherSource()
        assert accepts_adp_source(source) == "other"
