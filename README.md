# mobility-simulation

[![arXiv](https://img.shields.io/badge/arXiv-2311.11749-b31b1b.svg)](https://arxiv.org/abs/2311.11749)


## Install

Install the package in edit mode using:
```
pip install -e .
```

## Simulate

Run 

```
python example/run.py
```
for generating synthetic location traces. 



## Input and output

Required input files for Environment:
- Location visit sequence (empirical); csv file with columns `[user_id, location_id]`, row order shall follow visit sequence. Used for building the emperical markov matrix for IPT during preferential return. 
- Location with geometry and visitation freqency (empirical); csv file with columns `[id, geometry]`. `geometry` shall be in `wkt` format in latitude, longitude format in WGS1984 (`EPSG:4326` projection), e.g., `POINT (8.52244 47.38777)`. Used for calculating distanced between locations, and determining the attractiveness for each location for the density-EPR model during exploration. 

`./data/input/` folder contains demo data for running the simulation. 

Required parameters for simulation:
- wait time distribution: log-normal, power-law or truncated power law with corresponding parameters.
- jump length distribution: log-normal, power-law or truncated power law with corresponding parameters.
- Exploration parameters: gamma (normal distribution), rho (normal distribution) and p (default determined from gamma and rho). Setting p as a non-zero value will direct use p as exploration probability without using gamma and rho. 

Parameters determined from empirical data is stored in `example/config.yml` file.

Outputs location visit sequences with activity duration of a specified length (default 200) for a set of individuals (default 100). 

## Preprocessing from GNSS tracking dataset
We provide preprocessing script that includes necessary steps to transfer a GNSS tracking dataset into the required input file formats and obtain the parameters of empirical distributions for mobility simulation. We assume the raw tracking dataset contains stay points and triplegs (stages), and the processing script (`example/preprocess_tracking.py`) includes the following steps:
- Read staypoints and triplegs, and transforms them into trackintel compatible format.
- Calculate the tracking quality per user.
- Filter user with a minimum tracking quality. 
- Include only the records that occur within a geographical boundary (requires boundary shp). 
- Generate locations from staypoints
- Merge staypoints that occur close in time.
- Save locations and location transitions (input data for mobility simulation).
- Obtain the best fitting jump length distribution and wait time distribution (parameters for mobility simulation).
- Obtain the normal distribution parameters for gamma and rho (parameters for mobility simulation).

The generated locations and location transitions files will be save in the `./data/input/` folder.


## Known issues:
- This package requires geopandas dependency, which is best installed via conda-forge. 
- pref_return for ipt needs personal empirical markov matrix. Currently we maintain a single matrix for the whole dataset. For GC this is less of an issue, as GC individuals do not often share locations. 

## TODO:
None
