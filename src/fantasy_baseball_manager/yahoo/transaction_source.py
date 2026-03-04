import datetime
import logging
from typing import TYPE_CHECKING, Any

from fantasy_baseball_manager.domain import Transaction, TransactionPlayer
from fantasy_baseball_manager.yahoo.player_parsing import extract_player_data

if TYPE_CHECKING:
    from fantasy_baseball_manager.yahoo.client import YahooFantasyClient
    from fantasy_baseball_manager.yahoo.player_map import YahooPlayerMapper

logger = logging.getLogger(__name__)

_TYPE_MAP: dict[str, str] = {
    "add": "add",
    "drop": "drop",
    "add/drop": "add",
    "trade": "trade",
    "pending_trade": "trade",
    "waiver": "waiver",
}


class YahooTransactionSource:
    def __init__(self, client: YahooFantasyClient, mapper: YahooPlayerMapper) -> None:
        self._client = client
        self._mapper = mapper

    def fetch_transactions(
        self,
        league_key: str,
        *,
        since: datetime.datetime | None = None,
    ) -> list[tuple[Transaction, list[TransactionPlayer]]]:
        data = self._client.get_transactions(league_key)
        raw_transactions = self._parse_raw_transactions(data)

        result: list[tuple[Transaction, list[TransactionPlayer]]] = []
        for raw in raw_transactions:
            timestamp = datetime.datetime.fromtimestamp(int(raw["timestamp"]), tz=datetime.UTC)

            if since is not None and timestamp <= since:
                continue

            txn_type = self._normalize_type(raw.get("type", ""))
            status = raw.get("status", "successful")
            transaction_key = raw["transaction_key"]
            trader_team_key = raw.get("trader_team_key", "")
            tradee_team_key = raw.get("tradee_team_key")

            txn = Transaction(
                transaction_key=transaction_key,
                league_key=league_key,
                type=txn_type,
                timestamp=timestamp,
                status=status,
                trader_team_key=trader_team_key,
                tradee_team_key=tradee_team_key,
            )

            players = self._parse_players(raw, transaction_key)
            result.append((txn, players))

        return result

    @staticmethod
    def _parse_raw_transactions(data: dict[str, Any]) -> list[dict[str, Any]]:
        league_data = data.get("fantasy_content", {}).get("league")
        if not isinstance(league_data, list) or len(league_data) < 2:
            logger.warning("Unexpected transactions response structure")
            return []

        transactions = league_data[1].get("transactions")
        if not isinstance(transactions, dict):
            logger.warning("No transactions found in response")
            return []

        raw_list: list[dict[str, Any]] = []
        for key, value in transactions.items():
            if key == "count":
                continue
            txn_data = value.get("transaction")
            if not isinstance(txn_data, list) or len(txn_data) < 2:
                continue

            meta = txn_data[0]
            players_section = txn_data[1]

            raw: dict[str, Any] = {
                "transaction_key": meta.get("transaction_key", ""),
                "type": meta.get("type", ""),
                "timestamp": meta.get("timestamp", "0"),
                "status": meta.get("status", "successful"),
                "trader_team_key": meta.get("trader_team_key", ""),
                "tradee_team_key": meta.get("tradee_team_key"),
                "players": players_section,
            }
            raw_list.append(raw)
        return raw_list

    def _parse_players(self, raw: dict[str, Any], transaction_key: str) -> list[TransactionPlayer]:
        players_section = raw.get("players", {})
        if not isinstance(players_section, dict):
            return []

        players_data = players_section.get("players", players_section)
        if not isinstance(players_data, dict):
            return []

        result: list[TransactionPlayer] = []
        for key, value in players_data.items():
            if key == "count":
                continue
            player_info = value.get("player")
            if not isinstance(player_info, list) or not player_info:
                continue

            player_meta = player_info[0]
            player_data = extract_player_data(player_meta)

            # Extract transaction data for this player
            txn_data: dict[str, str] = {}
            for item in player_info:
                if isinstance(item, dict) and "transaction_data" in item:
                    td = item["transaction_data"]
                    if isinstance(td, list):
                        for td_item in td:
                            if isinstance(td_item, dict):
                                txn_data.update(td_item)
                    elif isinstance(td, dict):
                        txn_data = td

            # Resolve player via mapper
            mapping = self._mapper.resolve(player_data) if player_data.get("player_key") else None
            player_id = mapping.player_id if mapping is not None else None

            player_type = txn_data.get("type", "add")
            source_team_key = txn_data.get("source_team_key")
            dest_team_key = txn_data.get("destination_team_key")

            result.append(
                TransactionPlayer(
                    transaction_key=transaction_key,
                    player_id=player_id,
                    yahoo_player_key=player_data.get("player_key", ""),
                    player_name=player_data.get("name", "Unknown"),
                    source_team_key=source_team_key,
                    dest_team_key=dest_team_key,
                    type=player_type,
                )
            )

        return result

    @staticmethod
    def _normalize_type(raw_type: str) -> str:
        return _TYPE_MAP.get(raw_type, raw_type)
