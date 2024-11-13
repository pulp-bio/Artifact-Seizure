Copyright (C) 2024 ETH Zurich, Switzerland. SPDX-License-Identifier: Apache-2.0. See LICENSE file for details.

Authors: Thorir Mar Ingolfsson, Simone Benatti, Xiaying Wang, Adriano Bernini, Pauline Ducouret, Philippe Ryvlin, Sandor Beniczky, Luca Benini & Andrea Cossettini 
# EEG Analysis: Artifact and Seizure Detection

This repository contains two major components for EEG analysis: **Artifact Detection** and **Seizure Detection**. Each component includes code, data processing scripts, and experimental environments designed to reproduce results reported in associated research papers.

---

## Contents

- [Artifact Detection](#artifact-detection)
- [Seizure Detection](#seizure-detection)
- [Citing this Work](#citing-this-work)
- [License](#license)

---

## Artifact Detection

The **Artifact Detection** module provides an environment for detecting artifacts in EEG data using tree-based classifiers. The details of this approach are discussed in the papers:

1. *Minimizing artifact-induced false-alarms for seizure detection in wearable EEG devices with gradient-boosted tree classifiers*, available on [Nature](https://www.nature.com/articles/s41598-024-52551-0).
2. *Energy-efficient tree-based EEG artifact detection*, available on [IEEE](https://ieeexplore.ieee.org/document/9871413).

This module is located in the `Artifact_Detection` directory and includes scripts for data processing, model training, and evaluation.

### Getting Started with Artifact Detection

- **Dataset**: Download the TUH EEG Artifact Corpus (TUAR) from [here](https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml) (use version 2.0.0). Configure the path in `data_making/create_dataset.py` as specified in `Artifact_Detection/README.md`.
- **Environment Setup**: Create the required environment:
```
conda env create -f Artifact_Detection/TPOT.yml -n TPOT 
conda activate TPOT
```


- **Running the Code**:
- First, generate the datasets by running `create_dataset.py` in the `data_making` folder.
- Follow up with `make_features.py` to extract features and `unroll_data.py` to prepare the data.
- Train models using `TPOT_search.py` and evaluate using scripts in the `plotting` directory.

For more details, refer to `Artifact_Detection/README.md`.

---

## Seizure Detection

The **Seizure Detection** module is designed to minimize false alarms for seizure detection in wearable EEG devices using gradient-boosted tree classifiers. Details can be found in the paper *Minimizing artifact-induced false-alarms for seizure detection in wearable EEG devices with gradient-boosted tree classifiers*, available on [Nature](https://www.nature.com/articles/s41598-024-52551-0).

This module is located in the `Seizure_Detection` directory and includes all scripts for data preparation, model training, and evaluation.

### Getting Started with Seizure Detection

- **Dataset**: Download the CHB-MIT dataset from [here](https://physionet.org/content/chbmit/1.0.0/) and configure paths for `data_dir` and `data_save` in `make_data.py`.
- **Environment Setup**: Create the required environment:
```
conda env create -f Seizure_Detection/XGBoost.yml -n XGBoost 
conda activate XGBoost
```

- **Running the Code**:
- Generate dataset features by running `make_data.py`.
- Train models using `train_xgboost.py` for global models and cross-validation scripts (`train_xgboost_LOOCV.py`, `train_xgboost_RFCV.py`, and `train_xgboost_WFCV.py`) for subject-specific models.
- View results with the `read_results.ipynb` notebook.

For more information, see `Seizure_Detection/README.md`.

---

## Citing this Work

If you find this work useful, please cite the respective papers for each component:

For **Seizure Detection**:
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
For **Artifact Detection**:
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


---

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
