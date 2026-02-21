import io

from flask import Flask, Response

from fantasy_baseball_manager.domain.adp import ADP
from fantasy_baseball_manager.domain.league_settings import LeagueSettings
from fantasy_baseball_manager.domain.valuation import Valuation
from fantasy_baseball_manager.services.draft_board import build_draft_board, export_html


def create_live_draft_app(
    valuations: list[Valuation],
    league: LeagueSettings,
    player_names: dict[int, str],
    adp: list[ADP] | None = None,
) -> Flask:
    """Create a Flask app for live draft tracking.

    The app maintains a set of drafted player IDs. GET / rebuilds the board
    excluding drafted players and serves auto-refreshing HTML. POST /draft/<id>
    marks a player as drafted. POST /reset clears the drafted set.
    """
    app = Flask(__name__)

    valid_ids = {v.player_id for v in valuations}
    drafted_ids: set[int] = set()

    @app.route("/")
    def index() -> Response:
        remaining = [v for v in valuations if v.player_id not in drafted_ids]
        remaining_names = {pid: name for pid, name in player_names.items() if pid not in drafted_ids}
        board = build_draft_board(remaining, league, remaining_names, adp=adp)
        buf = io.StringIO()
        export_html(board, league, buf, auto_refresh=5)
        return Response(buf.getvalue(), mimetype="text/html")

    @app.route("/draft/<int:player_id>", methods=["POST"])
    def draft_player(player_id: int) -> tuple[Response, int]:
        if player_id not in valid_ids:
            return Response(f"Player {player_id} not found", mimetype="text/plain"), 404
        drafted_ids.add(player_id)
        return Response("OK", mimetype="text/plain"), 200

    @app.route("/reset", methods=["POST"])
    def reset() -> tuple[Response, int]:
        drafted_ids.clear()
        return Response("OK", mimetype="text/plain"), 200

    return app
