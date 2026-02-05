"""ADP (Average Draft Position) module for fantasy baseball."""

from fantasy_baseball_manager.adp.composite import (
    CompositeADPDataSource,
    CompositeADPSource,
    create_composite_adp_source,
)
from fantasy_baseball_manager.adp.espn_scraper import (
    ESPNADPDataSource,
    ESPNADPScraper,
    create_espn_adp_source,
)
from fantasy_baseball_manager.adp.models import ADPData, ADPEntry
from fantasy_baseball_manager.adp.protocol import ADPSource
from fantasy_baseball_manager.adp.registry import (
    get_datasource,
    get_source,
    list_datasources,
    list_sources,
    register_datasource,
    register_source,
)
from fantasy_baseball_manager.adp.scraper import (
    YahooADPDataSource,
    YahooADPScraper,
    create_yahoo_adp_source,
)

__all__ = [
    "ADPData",
    "ADPEntry",
    "ADPSource",
    "CompositeADPDataSource",
    "CompositeADPSource",
    "ESPNADPDataSource",
    "ESPNADPScraper",
    "YahooADPDataSource",
    "YahooADPScraper",
    "create_composite_adp_source",
    "create_espn_adp_source",
    "create_yahoo_adp_source",
    "get_datasource",
    "get_source",
    "list_datasources",
    "list_sources",
    "register_datasource",
    "register_source",
]
