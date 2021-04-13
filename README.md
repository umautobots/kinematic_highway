# A Kinematic Model for Trajectory Prediction in General Highway Scenarios
by Cyrus Anderson at [UM FCAV](https://fcav.engin.umich.edu/)

### Introduction
This paper presents a novel kinematic model to predict vehicles' trajectories in general
scenarios, including simultaneous lane changes by multiple vehicles and heavy occlusions.
More details are given in the preprint at https://arxiv.org/abs/2103.16673.

### Datasets
The NGSIM and highD datasets are used to evaluate the method, whose root folder should be set in `utils.py`.
The default setup uses `datasets` as a symbolic link:
```
baselines/
datasets/
|__tt_format
    |__10hz
         |__ngsim/
             |__i80/
             |__us101/
         |__highd/
|__ngsim/
    |__i-80/
    |__us-101/
|__highd-dataset-v1.0/
```
where the raw datasets (NGSIM and highD) are converted to `tt_format` for evaluation.
Conversion uses the tools in `loading_utils/dataset_conversion.py`.
The un-formatted `ngsim` folder contains the NGSIM dataset ([dataset portal](https://data.transportation.gov/Automobiles/Next-Generation-Simulation-NGSIM-Vehicle-Trajector/8ect-6jqj)
and [homepage](https://ops.fhwa.dot.gov/trafficanalysistools/ngsim.htm)).
The `highd-dataset-v1.0` folder contains the highD dataset ([homepage](https://www.highd-dataset.com/)).

(Note: There may be small errors when first loading the NGSIM data due to
small inconsistencies between the folder name formats/column names
of the US-101 and I-80 datasets - manually changing them can solve this.)

### Predict Trajectories
Predictions can be made by running
```
python driver.py
```

### Dependencies

- NumPy
- SciPy
- pandas
- matplotlib
