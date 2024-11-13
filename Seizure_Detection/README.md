Copyright (C) 2024 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file at the root of the repository for details.

Authors: Thorir Mar Ingolfsson, Simone Benatti, Xiaying Wang, Adriano Bernini, Pauline Ducouret, Philippe Ryvlin, Sandor Beniczky, Luca Benini & Andrea Cossettini 

# Seizure Detection
This project provides the experimental environment used to produce the results reported in the paper *Minimizing artifact-induced false-alarms for seizure detection in wearable EEG devices with gradient-boosted tree classifiers* available on [Nature](https://www.nature.com/articles/s41598-024-52551-0). If you find this work useful in your research, please cite
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
## Getting started

### Prerequisites

* This code is based on Python 3, and [Anaconda3](https://www.anaconda.com/distribution/) is required.
* You will need to download the CHB-MIT dataset, which is available [here](https://physionet.org/content/chbmit/1.0.0/). Once downloaded, configure `data_dir` and `data_save` in `make_data.py` (and other relevant scripts) to point to the dataset and the directory where you would like to save processed features.

### Installing
Navigate to the main folder and create the environment using Anaconda:
```
conda env create -f XGBoost.yml -n XGBoost 
conda activate XGBoost
```
Note: The environment file (`XGBoost.yml`) was created on a Linux machine, so it might need modifications for compatibility with other operating systems.

## Usage

**Generate Dataset Features**:  
First and foremest the dataset needs to be created. This is done by running the make_data.py file. This will generate the required feqtures from the raw EEG data. Both subject specific features and global features. Do this by running:
```
python make_data.py
```
Ensure that `data_dir` and `data_save` in `make_data.py` and other scripts point to the CHB-MIT dataset location and the desired output directory. For example, `data_dir` might look like: `path_to_downloaded_dataset/chbmit/1.0.0/`.

**Train Models**:

- **Global Model**:  
  Run the following command to train an XGBoost model on global features:
  ```
  python train_xgboost.py
  ```

- **Subject-Specific Model**:  
  Run the following commands to train XGBoost models on subject-specific features with different cross-validation methods:
  ```
  python train_xgboost_LOOCV.py    # Leave-One-Out Cross-Validation
  python train_xgboost_RFCV.py     # Rolling Window Cross-Validation
  python train_xgboost_WFCV.py     # Walk-Forward Cross-Validation
  ```

## Results

To view the results, open and run the `read_results.ipynb` notebook. This notebook will load the results from each model and cross-validation method and display a summary table with relevant evaluation metrics.

## License and Attribution

This code is licensed under the Apache License 2.0. Please refer to the LICENSE file at the root of the repository for details.
