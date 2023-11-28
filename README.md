# Individual mobility simulation models

[![arXiv](https://img.shields.io/badge/arXiv-2311.11749-b31b1b.svg)](https://arxiv.org/abs/2311.11749)


## Requirements, dependencies and installation
This code has been tested on

- Python 3.10, trackintel 1.2.2, geopandas 0.13.2

To create a virtual environment and install the required dependencies, please run the following:
```shell
git clone https://github.com/irmlma/mobility-simulation.git
cd mobility-simulation
conda env create -f environment.yml
conda activate mobsim
```
in your working folder. You can then install the package in edit mode using:
```
pip install -e .
```

## Simulate

Run 

```
python example/run.py
```
for generating synthetic location traces. We support epr, ipt, density-epr and dt-epr models. Outputs location visit sequences with activity duration of a specified length (default 200 steps) for a set of individuals (default 100). 

## Input and output

Required input files for running the generation, ideally obtained from empirical tracking dataset:

- Location with geometry and visitation freqency: csv file with columns `[id, geometry]`. `geometry` shall be in `wkt` format in latitude, longitude format in WGS1984 (`EPSG:4326` projection), e.g., `POINT (8.52244 47.38777)`. Required for calculating distanced between locations, and determining the attractiveness for each location for the density-EPR model during exploration. 
- Location visit sequence: csv file with columns `[user_id, location_id]`, row order shall follow the visited sequence. Required for building the emperical markov matrix for the IPT model during preferential return. 

Required parameters for simulation:
- Wait time distribution: log-normal, power-law or truncated power law with corresponding parameters.
- Jump length distribution: log-normal, power-law or truncated power law with corresponding parameters.
- Exploration parameters: gamma (normal distribution), rho (normal distribution) and p (default determined from gamma and rho). Setting p as a non-zero value will direct use p as exploration probability without using gamma and rho. 

Default parameters are determined from the SBB Green Class (GC) dataset, and are stored in `example/config.yml` file.

### Synthetic dataset

`./data/input/` folder contains demo data for running the simulation, which is generated using the trajectories from GC dataset. Specifically, locations from the GC dataset are projected into the level 13 grid of s2geometry, and the location transition sequence is obtained through forward simulation using the provided DT-EPR model. 


## Preprocessing from GNSS tracking dataset

We provide preprocessing script that includes necessary steps to transfer a GNSS tracking dataset into required input file formats and obtain parameters of empirical distributions for mobility simulation. We assume the raw tracking dataset contains stay points and triplegs (also called stages, representing continuous movement without changing mode, vehicle or stopping), and the processing script (`example/preprocess_tracking.py`) includes the following steps:
- Read staypoints and triplegs, and transforms them into trackintel compatible format.
- Calculate the temporal tracking quality per user.
- Filter user: include only users with sufficient high tracking quality. 
- Include only the records that occur within a geographical boundary (requires boundary shp). 
- Generate locations from staypoints. 
- Merge staypoints that occur close in time and belong to the same location.
- Save locations and location transitions (input data for mobility simulation).
- Obtain the best fitting jump length distribution and wait time distribution (parameters for mobility simulation).
- Obtain the normal distribution parameters for gamma and rho (parameters for mobility simulation).

The generated locations and location transitions files will be save in the `./data/input/` folder.

## Known issues:
- pref_return for ipt needs personal empirical markov matrix. Currently we maintain a single matrix for the whole dataset. For GC this is less of an issue, as GC individuals do not often share locations. 

## TODO:
None

## Citation
If you find this code useful for your work or use it in your project, please consider citing:

```shell
@misc{hong_revealing_2023,
    title={Revealing behavioral impact on mobility prediction networks through causal interventions},
    author={Hong, Ye and Xin, Yanan and Dirmeier, Simon and Perez-Cruz, Fernando and Raubal, Martin},
    publisher={arXiv},
    year={2023},
    url = {https://arxiv.org/abs/2311.11749},
    doi = {10.48550/arXiv.2311.11749},
}
```

## Contact
If you have any questions, open an issue or let me know: 
- Ye Hong {hongy@ethz.ch}

