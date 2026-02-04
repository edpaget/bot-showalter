# Design: Physics-Based Park Factor Validation

## Overview

Use pybaseball's physics-based batted ball trajectory simulation combined with actual stadium dimensions to compute park factors from first principles. This approach models how the same batted ball would behave differently in each of the 30 MLB parks based on elevation, weather, and fence geometry.

This is intended as an offline validation and analysis tool, not a primary data source for the projection pipeline.

## Motivation

- Validate empirically-derived park factors (from Savant or custom Statcast computation) against physics predictions
- Model parks that have changed dimensions or installed humidors (e.g., Coors humidor 2002, Arizona humidor 2023)
- Predict park effects for hypothetical scenarios (new stadiums, rule changes)
- Decompose WHY a park plays the way it does: elevation vs. dimensions vs. wind vs. humidity

## Components

### BattedBallTrajectory Simulator

`pybaseball.analysis.trajectories.batted_balls.calculator.BattedBallTrajectory` implements Alan Nathan's physics model of ball flight through air. It accepts:

- **Initial conditions:** exit velocity, launch angle, spray angle, spin rate
- **EnvironmentalParameters:**
  - `elevation_ft` (default 15) -- the dominant factor; Coors Field sits at 5,280 ft
  - `temperature_f` (default 70)
  - `pressure_in_hg` (default 29.92)
  - `relative_humidity` (default 50)
  - `vwind`, `phiwind`, `hwind` -- wind speed, direction, and measurement height

The simulator returns a full trajectory as (x, y, z) coordinates over time.

### Stadium Geometry

`pybaseball.plotting.STADIUM_COORDS`, loaded from `pybaseball/data/mlbstadiums.csv`, provides outfield wall coordinates for all 30 MLB parks in the Statcast coordinate system (matching `hc_x`/`hc_y` from pitch-level data). These can be used to determine whether a simulated trajectory clears the fence at a given spray angle.

### Environmental Data by Park

The following data must be compiled for each venue:

| Parameter | Source | Notes |
|-----------|--------|-------|
| Elevation (ft) | Static reference data | Biggest single factor; ranges from ~0 (MIA) to 5,280 (COL) |
| Average game-time temperature | Weather APIs or historical records | Varies by month; April games vs. August games |
| Barometric pressure | Derived from elevation + weather | Correlated with elevation but varies day-to-day |
| Humidity | Historical weather data | Higher humidity slightly reduces air density |
| Wind speed/direction | Most difficult; highly variable per-game | Wrigley Field is the canonical example of wind impact |

## Methodology

1. Define a representative sample of batted balls as a grid:
   - Exit velocity: 85--115 mph in 1 mph increments
   - Launch angle: 10--45 degrees in 1 degree increments
   - Spray angle: -45 to +45 degrees in 5 degree increments
2. For each batted ball in the sample, simulate the trajectory in each of the 30 parks using that park's environmental parameters
3. Check whether the trajectory clears the outfield fence at the appropriate spray angle using `STADIUM_COORDS` geometry
4. Compute HR probability for each park as the fraction of the sample that clears the fence
5. Normalize to league average = 1.0 to produce an HR park factor
6. For non-HR hit types, use the landing point relative to wall distance and outfield depth to estimate extra-base hit probability (less precise than for HR)

### Weighting the Sample

The grid should be weighted by the empirical frequency of each EV/LA/spray bucket in real Statcast data. A uniform grid would over-represent rare batted ball profiles. Pull one season of `statcast()` data, bin by EV/LA/spray, and use the bin counts as weights.

## Use Cases

### Validation
Compare simulated HR factors against empirical Savant factors. Large discrepancies suggest factors not captured by the model (wind patterns, altitude-adjusted ball behavior, outfield surface).

### Humidor Modeling
Simulate pre- and post-humidor installation by adjusting the coefficient of restitution or the environmental humidity parameter. Arizona installed a humidor in 2023; comparing simulated vs. actual factor changes validates the model.

### New Stadium Prediction
Estimate park factors before any games are played by inputting the stadium's proposed dimensions (as wall coordinates) and location (for elevation and climate).

### Factor Decomposition
For Coors Field, run simulations with:
- Coors elevation + Coors dimensions = full effect
- Sea-level elevation + Coors dimensions = fence-only effect
- Coors elevation + league-average dimensions = elevation-only effect

The difference quantifies how much of the Coors factor comes from each source.

## Limitations

- **Wind is highly variable.** Per-game wind data is not reliably available at scale. Average wind direction is a rough proxy, and some parks (Wrigley) have extreme day-to-day variation.
- **Ground ball effects are not modeled.** Turf vs. grass, drainage, infield dirt, and outfield wall angles affect singles, doubles, and triples but are not captured by ball flight physics.
- **Spin rate data on batted balls is limited.** Statcast does not publish batted ball spin reliably, so the model must assume a default spin rate.
- **Fielding is ignored.** Outfielder positioning, wall padding, warning track surface, and foul territory dimensions affect outcomes but are outside the scope of trajectory simulation.
- **Computational cost.** Simulating ~11,000+ trajectories (31 EV x 36 LA x 19 spray) across 30 parks produces ~330,000 trajectory calculations. This is feasible but not instantaneous; caching results is important.

## Integration

- Not intended as a `ParkFactorProvider` implementation for the pipeline
- Run as an offline script or notebook for analysis
- Store results as reference data (JSON or CSV) for comparison against empirical providers
- Could be triggered periodically (e.g., once per season) or when a park changes dimensions
- Flag parks where simulated and empirical factors diverge by more than a threshold (e.g., 5%) for manual investigation

## References

- Alan Nathan, "Trajectory Calculator" (physics.illinois.edu)
- pybaseball `BattedBallTrajectory` implementation in `pybaseball.analysis.trajectories.batted_balls.calculator`
- Robert Arthur, "How the New Humidor Could Change Baseball in Arizona" (FiveThirtyEight)
- Nathan, Alan M. "The Physics of Baseball" (University of Illinois)
