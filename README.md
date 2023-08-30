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

Required input files for Environment:
- Top location visited by each user (empirical); csv file with columns `[user_id, location_id]`. Used for assigning simulated users to their initial location.
- Location visit sequence (empirical); csv file with columns `[user_id, location_id]`, row order shall follow visit sequence. Used for assigning simulated users to their initial location. Used for building the emperical markov matrix for IPT during preferential return. 
- Location with geometry and visitation freqency (empirical); csv file with columns `[id, center, count]`. `center` shall be in `wkt` format in `EPSG:2056` projection, e.g., `POINT (2625338.71 1229204.85)`. Used for calculating distanced between locations, and determining the attractiveness for each location for the density-EPR model during exploration. 

Parameters determined from empirical data is stored in `example/config.yml` file. Jump length and Wait time follows log-normal distribution, and rho and gamma follows normal distribution. 

## Known issues:
- this package requires geopandas dependency, which is best installed via conda-forge; thus I included an environment file. I do not know how to define this in setup.py
- pref_return for ipt needs personal empirical markov matrix. Currently we maintain a single matrix for the whole dataset. For GC this is less of an issue, as GC individuals do not often share locations. 
- only `mobsim/df_epr.py` and `mobsim/env.py` is implemented.