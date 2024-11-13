#*----------------------------------------------------------------------------*
#* Copyright (C) 2024 ETH Zurich, Switzerland                                 *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Thorir Mar Ingolfsson                                             *
#*----------------------------------------------------------------------------*

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
import os
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from utils.metrics_EEG import train_and_evaluate_model


# Define paths for data directories (to be set based on user environment)
data_dir = "#CHANGEME"
data_save = "#CHANGEME"

# Load and sort patient directories
patients = os.listdir(data_dir)
patients.sort()
#only read the diretories that start with 'chb'
patients = [p for p in patients if p.startswith('chb')]
patients.pop()


# Configuration for parallel processing and model training
n_cores = 2
random_state = 42
complexity = 100
list_weight = [i for i in range(1,500)]
list_weight.insert(0,1)

# Create results directory if it doesn't exist
os.makedirs("results/subject_specific/", exist_ok=True)


window_size = 3

for sec in [1, 2, 4, 8]:
    print("Processing ", sec, " second files")
    for patient in patients:
        print("Processing patient: ", patient)
        # Loop over all the seizure files for this patient
        files = os.listdir(data_save + patient)
        files.sort()
        
        # Determine the trials available
        if(patient == 'chb17'):
            trials = [file[7:9] for file in files if file.endswith('features_1s.npy') and file.startswith(patient)]
        else:
            trials = [file[6:8] for file in files if file.endswith('features_1s.npy') and file.startswith(patient)]
        num_trials = len(trials)
        
        # Start validation from the second trial onwards
        for i in range(1, num_trials):
            if not os.path.exists(f"results/subject_specific/{patient}/{trials[i]}_xgboost_rolling_{sec}s.pkl"):
                cnt = 0
                
                # Collect training data from the most recent `window_size` trials before the validation trial
                for j in range(max(0, i - window_size), i):
                    if cnt == 0:
                        if patient == 'chb17':
                            if trials[j] == '63':
                                X_train = np.load(data_save + patient + '/' + patient + 'b_' + trials[j] + f'_features_{sec}s.npy')
                                y_train = np.load(data_save + patient + '/' + patient + 'b_' + trials[j] + f'_labels_{sec}s.npy')
                            else:
                                X_train = np.load(data_save + patient + '/' + patient + 'a_' + trials[j] + f'_features_{sec}s.npy')
                                y_train = np.load(data_save + patient + '/' + patient + 'a_' + trials[j] + f'_labels_{sec}s.npy')
                        else:
                            X_train = np.load(data_save + patient + '/' + patient + '_' + trials[j] + f'_features_{sec}s.npy')
                            y_train = np.load(data_save + patient + '/' + patient + '_' + trials[j] + f'_labels_{sec}s.npy')
                        cnt += 1
                    else:
                        if patient == 'chb17':
                            if trials[j] == '63':
                                X_train = np.concatenate((X_train, np.load(data_save + patient + '/' + patient + 'b_' + trials[j] + f'_features_{sec}s.npy')))
                                y_train = np.concatenate((y_train, np.load(data_save + patient + '/' + patient + 'b_' + trials[j] + f'_labels_{sec}s.npy')))
                            else:
                                X_train = np.concatenate((X_train, np.load(data_save + patient + '/' + patient + 'a_' + trials[j] + f'_features_{sec}s.npy')))
                                y_train = np.concatenate((y_train, np.load(data_save + patient + '/' + patient + 'a_' + trials[j] + f'_labels_{sec}s.npy')))
                        else:
                            X_train = np.concatenate((X_train, np.load(data_save + patient + '/' + patient + '_' + trials[j] + f'_features_{sec}s.npy')))
                            y_train = np.concatenate((y_train, np.load(data_save + patient + '/' + patient + '_' + trials[j] + f'_labels_{sec}s.npy')))

                # Load the validation data
                if patient == 'chb17':
                    if trials[i] == '63':
                        X_test = np.load(data_save + patient + '/' + patient + 'b_' + trials[i] + f'_features_{sec}s.npy')
                        y_test = np.load(data_save + patient + '/' + patient + 'b_' + trials[i] + f'_labels_{sec}s.npy')
                    else:
                        X_test = np.load(data_save + patient + '/' + patient + 'a_' + trials[i] + f'_features_{sec}s.npy')
                        y_test = np.load(data_save + patient + '/' + patient + 'a_' + trials[i] + f'_labels_{sec}s.npy')
                else:
                    X_test = np.load(data_save + patient + '/' + patient + '_' + trials[i] + f'_features_{sec}s.npy')
                    y_test = np.load(data_save + patient + '/' + patient + '_' + trials[i] + f'_labels_{sec}s.npy')
                
                # Now we have the training and testing data for this trial.
                # Train an xgboost model on this data.
                result = Parallel(n_jobs=n_cores)(delayed(train_and_evaluate_model)(X_train, y_train, random_state, X_test, y_test, weight, complexity, int(sec), 'logloss') for weight in list_weight)
                df = pd.DataFrame(result)
                df.to_pickle(f"results/subject_specific/{patient}/{trials[i]}_xgboost_rolling_{sec}s.pkl")