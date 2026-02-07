"""ADP (Average Draft Position) module for fantasy baseball."""

from fantasy_baseball_manager.adp.composite import (
    CompositeADPDataSource,
    create_composite_adp_source,
)
from fantasy_baseball_manager.adp.espn_scraper import (
    ESPNADPDataSource,
    create_espn_adp_source,
)
from fantasy_baseball_manager.adp.models import ADPData, ADPEntry
from fantasy_baseball_manager.adp.registry import (
    get_datasource,
    list_datasources,
    register_datasource,
)
from fantasy_baseball_manager.adp.scraper import (
    YahooADPDataSource,
    create_yahoo_adp_source,
)

__all__ = [
    "ADPData",
    "ADPEntry",
    "CompositeADPDataSource",
    "ESPNADPDataSource",
    "YahooADPDataSource",
    "create_composite_adp_source",
    "create_espn_adp_source",
    "create_yahoo_adp_source",
    "get_datasource",
    "list_datasources",
    "register_datasource",
]
