Copyright (C) 2024 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

Authors: Thorir Mar Ingolfsson, Simone Benatti, Xiaying Wang, Adriano Bernini, Pauline Ducouret, Philippe Ryvlin, Sandor Beniczky, Luca Benini & Andrea Cossettini 

# Artifact Detection
This project provides the experimental environment used to produce the results reported in the papers *Minimizing artifact-induced false-alarms for seizure detection in wearable EEG devices with gradient-boosted tree classifiers* and *Energy-efficient tree-based EEG artifact detection* available on [Nature](https://www.nature.com/articles/s41598-024-52551-0) and [IEEE](https://ieeexplore.ieee.org/document/9871413). If you find this work useful in your research, please cite
```
@article{ingolfsson2024minimizing,
  title={Minimizing artifact-induced false-alarms for seizure detection in wearable EEG devices with gradient-boosted tree classifiers},
  author={Ingolfsson, Thorir Mar and Benatti, Simone and Wang, Xiaying and Bernini, Adriano and Ducouret, Pauline and Ryvlin, Philippe and Beniczky, Sandor and Benini, Luca and Cossettini, Andrea},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={2980},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
```
and
```
@inproceedings{ingolfsson2022energy,
  title={Energy-efficient tree-based EEG artifact detection},
  author={Ingolfsson, Thorir Mar and Cossettini, Andrea and Benatti, Simone and Benini, Luca},
  booktitle={2022 44th Annual International Conference of the IEEE Engineering in Medicine \& Biology Society (EMBC)},
  pages={3723--3728},
  year={2022},
  organization={IEEE}
}

```

## Getting started

### Prerequisites
* The code is based on Python3, and [Anaconda3](https://www.anaconda.com/distribution/) is required.

Also the dataset TUH EEG Artifact Corpus (TUAR) needs to be downloaded and put on the machine. It is available on [here](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml), we make use of the V2.0.0 of the dataset.
Please modify then the path of 'PATH_CHANGE' inside 'data_making/create_dataset.py' to the path of the downloaded dataset.

We also make use of the NEDC PyPrint EDF (PYPR) library which can be found [here](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) and it needs to be downloaded and extracted into the utils/NEDC/ folder of the project.

### Installing
Navigate to the main folder and create the environment using Anaconda:
```
conda env create -f TPOT.yml -n TPOT
conda activate TPOT
```
Note: The environment file (`TPOT.yml`) was created on a Linux machine, so it might need modifications for compatibility with other operating systems.

## Usage
First and foremest the dataset needs to be created. This is done by running the create_dataset.py script. This script will create the dataset and save it in the data folder. 
Please note that the dataset creation script takes in two arguments, first one being the frequencies (must be must be in class [250, 250_1000, 256, 256_512, all]) and the second one is the mode must be in class [binary, multi_binary, multioutput]. To get all the datasets the script needs to be run with the following commands:
```
cd data_making
python create_dataset.py 250 binary
python create_dataset.py 250 multi_binary
python create_dataset.py 250 multioutput
....
python create_dataset.py all binary
python create_dataset.py all multi_binary
python create_dataset.py all multioutput
```
For all combinations of frequencies and modes (15 in total). Please note that the 'SAVE_PATH' needs to be changed to the path where the dataset should be saved.

Next the features need to be created which is done by running the make_features.py script, which takes in one arguments the frequencies (must be must be in class [250, 250_1000, 256, 256_512, all]). An example of running the script is: (after also changing the 'SAVE_PATH' to the path where the features should be saved)
```
python make_features.py 250
```
After making features for all features run the unroll_data.py script to unroll the feature and labels and make sure the labels are correctly saved. this is done by running:
```
python unroll_data.py
```

After the dataset is created the TPOT can be run. This is done by running the TPOT.py script. This script will run the TPOT algorithm on the dataset and save the best pipeline in the models folder. The script takes in 6 arguments, the first one being the frequencies (must be must be in class [250, 250_1000, 256, 256_512, all]) and the second one is the mode must be in class [BI, MB, MO] for binary, multibinary and multioutput correspendingly. The other four arguments are the number of generations, population size, the number of folds for cross validation and if to use unrolled features and labels. To run the script for one of the datasets the following commands need to be run:
```
python TPOT_search.py 250 BI 100 100 5 False
```
This would run the TPOT search algorithm for the 250Hz dataset with binary mode for 100 generations, population size of 100, 5 folds for cross validation and not using unrolled features and labels.

To see an example of the extracted pipelines for different runs of TPOT_search.py please navigate to the exported_pipelines folder. 

Next to verify the scores of each exported pipeline please go into the scores folder and ru nteh scores_for_all_datasets.py script. This script will print out the scores for each pipeline for each dataset.

We also provide code to reproduce the plotting of GradientBoostingClassifier vs ExtraTreesClassifer, please navigate to the plotting folder and run the following commands to get the plots:
```
python gradient_for_plot.py
python mmc_gradient_plot.py
python mmc_extra_plot.py
python run_for_plot.py
python gradient_vs_extra.py
```
which will produce the plots for the GradientBoostingClassifier vs ExtraTreesClassifer inside the figures subfolder of plotting folder.

## License and Attribution
This code is licensed under the Apache License 2.0. Please refer to the LICENSE file at the root of the repository for details.

