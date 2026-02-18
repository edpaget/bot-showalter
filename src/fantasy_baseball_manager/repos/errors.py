from fantasy_baseball_manager.domain.player import Player
from fantasy_baseball_manager.exceptions import FbmException


class PlayerConflictError(FbmException):
    def __init__(self, new_player: Player, existing_player: Player, conflicting_column: str) -> None:
        self.new_player = new_player
        self.existing_player = existing_player
        self.conflicting_column = conflicting_column
        super().__init__(
            f"Player conflict on {conflicting_column}: "
            f"new player (mlbam_id={new_player.mlbam_id}) conflicts with "
            f"existing player (id={existing_player.id}, mlbam_id={existing_player.mlbam_id})"
        )
