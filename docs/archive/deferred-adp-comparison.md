# ADP Comparison Feature - DEFERRED

## Status
Feature deferred due to lack of accessible ADP data sources.

## Overview
The goal was to compare projected auction values or rankings against Average Draft Position (ADP) data to identify value opportunities in drafts.

## Research Summary

### APIs Investigated

| Source | Result |
|--------|--------|
| Yahoo Fantasy API | Only returns your league's draft results, not global ADP |
| FantasyData/SportsDataIO | Has ADP endpoint but requires paid subscription |
| pybaseball | No ADP support; projection PR (#335) still open/unmerged |

### Scraping Required (Not Pursued)

| Source | Issue |
|--------|-------|
| FanGraphs | NFBC ADP in projections pages, but data is JS-rendered |
| FantasyPros | ADP pages are React SPA, require JS execution |
| Yahoo ADP page | React-rendered, no direct data access |
| NFBC direct | Requires authentication, no public API |

### Manual CSV Option
FantasyData has CSV/XLS export buttons that could support a `--adp-file` flag for manual input, but this doesn't provide automated ADP comparison.

## Future Options

1. **Monitor pybaseball** - PR #335 may add projection functions that include ADP
2. **Paid subscription** - FantasyData subscription if feature becomes high priority
3. **Revisit scraping** - If tolerance for browser automation changes
4. **Yahoo API changes** - Check periodically if Yahoo adds ADP to their API

## Alternative Pre-Draft Improvements

Consider these alternatives that don't require external ADP data:

- Improved auction value calculations using existing projections
- Draft strategy recommendations based on positional scarcity
- Tier-based player groupings for draft boards
- League-specific settings optimization
