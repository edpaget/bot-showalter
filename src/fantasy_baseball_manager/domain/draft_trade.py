from dataclasses import dataclass


@dataclass(frozen=True)
class DraftTrade:
    team_a: int
    team_b: int
    team_a_gives: list[int]  # pick numbers team_a gives away
    team_b_gives: list[int]  # pick numbers team_b gives away
