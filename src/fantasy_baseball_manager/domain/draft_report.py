from dataclasses import dataclass


@dataclass(frozen=True)
class PickGrade:
    pick_number: int
    player_id: int
    player_name: str
    position: str
    value: float
    best_available_value: float
    grade: float


@dataclass(frozen=True)
class StealOrReach:
    pick_number: int
    player_id: int
    player_name: str
    position: str
    value: float
    pick_delta: int


@dataclass(frozen=True)
class CategoryStanding:
    category: str
    total_z: float
    rank: int
    teams: int


@dataclass(frozen=True)
class DraftReport:
    total_value: float
    optimal_value: float
    value_efficiency: float
    budget: int | None
    total_spent: int | None
    category_standings: list[CategoryStanding]
    pick_grades: list[PickGrade]
    mean_grade: float
    steals: list[StealOrReach]
    reaches: list[StealOrReach]
