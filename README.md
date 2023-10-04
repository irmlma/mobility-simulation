# mobility-simulation

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
- Top location visited by each user (empirical); csv file with columns `[user_id, location_id]`. Used for assigning simulated users to their initial location.
- Location visit sequence (empirical); csv file with columns `[user_id, location_id]`, row order shall follow visit sequence. Used for building the emperical markov matrix for IPT during preferential return. 
- Location with geometry and visitation freqency (empirical); csv file with columns `[id, center, count]`. `center` shall be in `wkt` format in `EPSG:2056` projection, e.g., `POINT (2625338.71 1229204.85)`. Used for calculating distanced between locations, and determining the attractiveness for each location for the density-EPR model during exploration. 

Required parameters for simulation:
- wait time distribution: log-normal, power-law or truncated power law with corresponding parameters.
- jump length distribution: log-normal, power-law or truncated power law with corresponding parameters.
- Exploration parameters: gamma (normal), rho (normal) and p (default determined from gamma and rho). Setting p as a non-zero value will direct use p as exploration probability without using gamma and rho. 
Parameters determined from empirical data is stored in `example/config.yml` file.

Outputs location visit sequences with activity duration of a specified length (default 20) for each individual (default 100). 


## Known issues:
- selection of users hard coded. 
- this package requires geopandas dependency, which is best installed via conda-forge; thus I included an environment file. I do not know how to define this in setup.py
- pref_return for ipt needs personal empirical markov matrix. Currently we maintain a single matrix for the whole dataset. For GC this is less of an issue, as GC individuals do not often share locations. 

## TODO:
- create demo data for open-source
    - define location as spatial grids
    - create dummy empirical visit data