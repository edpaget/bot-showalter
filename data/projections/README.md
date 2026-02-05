# Historical Projections

This directory contains historical projection CSV files for backtesting.

## File Naming Convention

Files should follow the pattern: `{system}_{year}_{batting|pitching}.csv`

Examples:
- `steamer_2023_batting.csv`
- `steamer_2023_pitching.csv`
- `zips_2022_batting.csv`
- `zips_2022_pitching.csv`

## Supported Systems

- `steamer` - Steamer projections
- `zips` - ZiPS projections
- `zipsdc` - ZiPS with Depth Charts playing time

## Downloading from FanGraphs

1. Go to [FanGraphs Projections](https://www.fangraphs.com/projections)
2. Select the projection system (Steamer, ZiPS, etc.)
3. For historical projections, you'll need a FanGraphs+ membership ($80/year)
4. Select the desired year
5. Click "Export Data" to download the CSV
6. Rename the file following the convention above

### Required Columns

**Batting:**
- `Name` - Player name
- `Team` - Team abbreviation
- `playerid` or `idfg` - FanGraphs player ID
- `MLBAMID` or `xMLBAMID` - MLB Advanced Media ID (optional)
- `PA`, `AB`, `H`, `2B`, `3B`, `HR`, `R`, `RBI`, `SB`, `CS`, `BB`, `SO`
- `1B` (optional - computed from H - 2B - 3B - HR if missing)
- `HBP`, `SF`, `SH` (optional)
- `OBP`, `SLG`, `OPS`, `wOBA`, `WAR` (optional)

**Pitching:**
- `Name` - Player name
- `Team` - Team abbreviation
- `playerid` or `idfg` - FanGraphs player ID
- `MLBAMID` or `xMLBAMID` - MLB Advanced Media ID (optional)
- `G`, `GS`, `IP`, `W`, `L`, `SV`, `HLD`, `SO`, `BB`, `H`, `ER`
- `HBP`, `HR` (optional)
- `ERA`, `WHIP` (optional - computed from stats if missing)
- `FIP`, `WAR` (optional)

## Usage

Once CSV files are placed in this directory, they will be automatically
detected and available as projection engines:

```bash
# List available engines (includes detected CSV projections)
uv run fantasy-baseball-manager engines list

# Evaluate historical projections
uv run fantasy-baseball-manager evaluate 2023 --engine steamer_2023 --engine marcel_gb

# Multi-year backtest
uv run fantasy-baseball-manager evaluate 2023 --engine steamer_2023 --years 2021,2022,2023
```

## Notes

- CSV files are not committed to the repository (ignored in .gitignore)
- FanGraphs historical projections require a paid membership
- Column names are case-insensitive
- Missing optional fields default to 0
