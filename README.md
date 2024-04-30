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
   [https://docs.python.org/3/library/venv.html](Python virtual environment), activate it and
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

- OpenStreetMap network imported: `metropy.road_network.osm`
- Road-network post-processing: `metropy.road_network.postprocess`
- METROPOLIS2 road-network writer: `metropy.road_network.write_metropolis_edges`
