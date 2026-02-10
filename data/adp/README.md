# FantasyPros Historical ADP Data

This directory stores historical ADP (Average Draft Position) CSV files downloaded from FantasyPros.

## File Naming Convention

```
fantasypros_{year}.csv
```

Examples: `fantasypros_2022.csv`, `fantasypros_2023.csv`, `fantasypros_2024.csv`

## Format

Comma-separated values with the following columns:

| Column    | Description                                      |
|-----------|--------------------------------------------------|
| Rank      | Overall rank                                     |
| Player    | Player name                                      |
| Team      | Team abbreviation                                |
| Positions | Comma-separated positions (e.g., `CF,RF,DH`)    |
| ESPN      | ESPN ADP                                         |
| RTS       | RTS ADP                                          |
| NFBC      | NFBC ADP                                         |
| FT        | FantasyTeam ADP                                  |
| CBS       | CBS ADP                                          |
| Yahoo     | Yahoo ADP                                        |
| AVG       | Composite average ADP (used as training target)  |

## Download Instructions

1. Log in to [FantasyPros](https://www.fantasypros.com/) (subscription required)
2. Navigate to MLB ADP for the desired year
3. Export as CSV
4. Save as `fantasypros_{year}.csv` in this directory

## Usage

```bash
uv run fantasy-baseball-manager ml build-dataset --years 2022,2023,2024,2025 --system steamer
```

These files are git-ignored since they contain subscription-gated data.
