"""Composite ADP source that aggregates from multiple providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from fantasy_baseball_manager.adp.models import ADPEntry
from fantasy_baseball_manager.adp.name_utils import normalize_name
from fantasy_baseball_manager.data.protocol import ALL_PLAYERS, DataSource, DataSourceError
from fantasy_baseball_manager.result import Err, Ok

if TYPE_CHECKING:
    from fantasy_baseball_manager.player.identity import Player


class CompositeADPDataSource:
    """Aggregates ADP from multiple DataSource[ADPEntry] sources.

    Implements the DataSource[ADPEntry] protocol by fetching data from
    multiple underlying sources, matching players by normalized name,
    and computing average ADP values.

    Usage:
        yahoo_source = create_yahoo_adp_source()
        espn_source = create_espn_adp_source()
        composite = CompositeADPDataSource([yahoo_source, espn_source])
        result = composite(ALL_PLAYERS)
    """

    def __init__(self, sources: list[DataSource[ADPEntry]]) -> None:
        """Initialize with a list of ADP data sources.

        Args:
            sources: List of DataSource[ADPEntry] instances to aggregate from.
        """
        self._sources = sources

    @overload
    def __call__(self, query: type[ALL_PLAYERS]) -> Ok[list[ADPEntry]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: list[Player]) -> Ok[list[ADPEntry]] | Err[DataSourceError]: ...

    @overload
    def __call__(self, query: Player) -> Ok[ADPEntry] | Err[DataSourceError]: ...

    def __call__(
        self, query: type[ALL_PLAYERS] | Player | list[Player]
    ) -> Ok[list[ADPEntry]] | Ok[ADPEntry] | Err[DataSourceError]:
        # Only ALL_PLAYERS queries are supported
        if query is not ALL_PLAYERS:
            return Err(DataSourceError("Only ALL_PLAYERS queries supported for composite ADP data"))

        # Collect all entries by normalized name
        # Each entry: (original_name, adp_values, positions, percent_drafted_values)
        aggregated: dict[str, tuple[str, list[float], set[str], list[float]]] = {}

        for source in self._sources:
            result = source(ALL_PLAYERS)
            if result.is_err():
                return result

            entries = result.unwrap()
            for entry in entries:
                norm_name = normalize_name(entry.name)

                if norm_name not in aggregated:
                    aggregated[norm_name] = (
                        entry.name,  # Preserve first seen name
                        [],
                        set(),
                        [],
                    )

                original_name, adp_values, positions, pct_drafted_values = aggregated[norm_name]
                adp_values.append(entry.adp)
                positions.update(entry.positions)
                if entry.percent_drafted is not None:
                    pct_drafted_values.append(entry.percent_drafted)

        # Build averaged entries
        result_entries: list[ADPEntry] = []
        for _norm_name, (original_name, adp_values, positions, pct_drafted_values) in aggregated.items():
            avg_adp = sum(adp_values) / len(adp_values)
            avg_pct = sum(pct_drafted_values) / len(pct_drafted_values) if pct_drafted_values else None

            result_entries.append(
                ADPEntry(
                    name=original_name,
                    adp=avg_adp,
                    positions=tuple(sorted(positions)),
                    percent_drafted=avg_pct,
                )
            )

        # Sort by ADP
        result_entries.sort(key=lambda e: e.adp)

        return Ok(result_entries)


def create_composite_adp_source(sources: list[DataSource[ADPEntry]]) -> DataSource[ADPEntry]:
    """Create a composite DataSource that aggregates from multiple ADP sources.

    Args:
        sources: List of DataSource[ADPEntry] instances to aggregate from.

    Returns:
        A CompositeADPDataSource that computes averaged ADP values.

    Usage:
        yahoo = create_yahoo_adp_source()
        espn = create_espn_adp_source()
        composite = create_composite_adp_source([yahoo, espn])
    """
    return CompositeADPDataSource(sources)
