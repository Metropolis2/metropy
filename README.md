# Python library for METROPOLIS2

This repository contains Python tools to work with METROPOLIS2 input and output data.

# Requirements

- Minimum supported Python version: `3.9`.

# How to use

1. Clone the repository:
   ```
   git clone https://github.com/Metropolis2/metropy
   ```
2. Install the dependencies listed in `requirements.txt`.
   The recommended way is to create a
   [Python virtual environment](https://docs.python.org/3/library/venv.html), activate it and
   install the dependencies on it:
   ```
   python -m venv venv/
   source venv/bin/activate # On MacOS / Linux
   venv\Scripts\Activate.ps1 # On Windows
   pip install -r requirements.txt
   ```
3. Change the variables inside `config.toml` to the desired paths and values.
4. Set the working directory of the terminal emulator to the directory where `config.toml` and
   `metropy/` are located.
5. Run the tools as Python modules.
   For example, to import a road-network from OpenStreetMap, use
   ```
   python -m metropy.road_network.osm
   ```

# Tools available

- OpenStreetMap network import: `metropy.road_network.osm`
- OpenStreetMap network import (for walking trips): `metropy.road_network.osm_walk`
- Road-network post-processing: `metropy.road_network.postprocess`
- Identify a restricted area on the road network: `metropy.road_network.restriction_area`
- METROPOLIS2 road-network writer: `metropy.road_network.write_metropolis_edges`
- Request routing queries from TomTom API: `metropy.calibration.tomtom_requests`
- Map matching of routing results to the road network: `metropy.calibration.map_matching`
- Post-processing for the map matching results: `metropy.calibration.post_map_matching`
- Free-flow travel time calibration: `metropy.calibration.free_flow_calibration.py`
- Edge capacities calibration: `metropy.calibration.capacities_calibration.py`
- [France synthetic population](https://github.com/eqasim-org/ile-de-france) import:
  `metropy.synthetic_population.france`
- Generating vehicles for the population from French vehicle fleet data:
  `metropy.synthetic_population.french_vehicle_fleet`
- Identifying the origin / destination zone of trips: `metropy.synthetic_population.french_zones`
- Predict mode choice for the synthetic population: `metropy.demand.predict_modes`
- Predict activities' desired start time and duration: `metropy.demand.desired_times`
- Draw mode-specific stochastic shocks: `metropy.demand.draw_mode_epsilons`
- Plotting a household from the synthetic population: `metropy.synthetic_population.plot_household`
- Generating a population from an origin-destination matrix: `metropy.od_matrix.disaggregate`
- Splitting road network in main and secondary parts: `metropy.routing.road_split`
- Walking distance computation: `metropy.routing.walking_distance`
- Public-transit routing with [OpenTripPlanner](http://www.opentripplanner.org/):
  `metropy.routing.opentripplanner`
- Run a simulation with only the road trips: `metropy.run.road_only`
- Run a simulation to retrieve the travel time of the TomTom requests from the results of a
  simulation: `metropy.run.tomtom_routes`
- Compute stop-to-stop public-transit flows: `metropy.public_transit.analyze_flows`
- Compute public-transit chevelus: `metropy.public_transit.chevelus`
- Compute pollutant emissions and fuel consumption with the EMISENS model:
  `metropy.emissions.emisens`
- Compare METROPOLIS2 free-flow travel times with travel survey data:
  `metropy.travel_survey.free_flow_travel_time`

# Acknowledgments

Many thanks to Kokouvi Joseph Djafon for his work on the calibration tools.
