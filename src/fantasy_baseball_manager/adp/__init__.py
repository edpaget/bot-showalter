"""ADP (Average Draft Position) module for fantasy baseball."""

from fantasy_baseball_manager.adp.composite import (
    CompositeADPDataSource,
    create_composite_adp_source,
)
from fantasy_baseball_manager.adp.csv_resolver import ADPCSVResolver
from fantasy_baseball_manager.adp.espn_scraper import (
    ESPNADPDataSource,
    create_espn_adp_source,
)
from fantasy_baseball_manager.adp.fantasypros_source import (
    FantasyProsADPDataSource,
    FantasyProsADPParser,
)
from fantasy_baseball_manager.adp.models import ADPData, ADPEntry
from fantasy_baseball_manager.adp.name_utils import normalize_name
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
    "ADPCSVResolver",
    "ADPData",
    "ADPEntry",
    "CompositeADPDataSource",
    "ESPNADPDataSource",
    "FantasyProsADPDataSource",
    "FantasyProsADPParser",
    "YahooADPDataSource",
    "create_composite_adp_source",
    "create_espn_adp_source",
    "create_yahoo_adp_source",
    "get_datasource",
    "list_datasources",
    "normalize_name",
    "register_datasource",
]
