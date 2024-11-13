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

for sec in [1,2,4,8]:
    print("Processing ", sec, " second files")
    for patient in patients:
        print("Processing patient: ", patient)
        # Loop over all the seizure files for this patient
        files = os.listdir(data_save + patient)
        files.sort()
        # files includes 1, 2, 4 and 8 second files + global files for them all.
        # We want to count how many files there are for each second.
        # list the different trials, they are named "chb01_xx_..."
        if(patient == 'chb17'):
            trials = [file[7:9] for file in files if file.endswith('features_1s.npy') and file.startswith(patient)]
        else:    
            trials = [file[6:8] for file in files if file.endswith('features_1s.npy') and file.startswith(patient)]
        num_trials = len(trials)
        # We always train on the first trial, but we next test on the second trial and third trial, etc.
        for i in range(num_trials-1):
            if(not os.path.exists("results/subject_specific/" + patient + "/" + trials[i+1] + "_xgboost_walk_"+str(sec)+"s.pkl")):
                for j in range(i+1):
                    if(j == 0):
                        if(patient == 'chb17'):
                            if(trials[j] == '63'):
                                X_train = np.load(data_save + patient + '/' + patient + 'b_' + trials[j] + '_features_'+str(sec)+'s.npy')
                                y_train = np.load(data_save + patient + '/' + patient + 'b_' + trials[j] + '_labels_'+str(sec)+'s.npy')
                            else:
                                X_train = np.load(data_save + patient + '/' + patient + 'a_' + trials[j] + '_features_'+str(sec)+'s.npy')
                                y_train = np.load(data_save + patient + '/' + patient + 'a_' + trials[j] + '_labels_'+str(sec)+'s.npy')
                        else:
                            X_train = np.load(data_save + patient + '/' + patient + '_' + trials[j] + '_features_'+str(sec)+'s.npy')
                            y_train = np.load(data_save + patient + '/' + patient + '_' + trials[j] + '_labels_'+str(sec)+'s.npy')
                    else:
                        if(patient == 'chb17'):
                            if(trials[j] == '63'):
                                X_train = np.concatenate((X_train, np.load(data_save + patient + '/' + patient + 'b_' + trials[j] + '_features_'+str(sec)+'s.npy')))
                                y_train = np.concatenate((y_train, np.load(data_save + patient + '/' + patient + 'b_' + trials[j] + '_labels_'+str(sec)+'s.npy')))
                            else:
                                X_train = np.concatenate((X_train, np.load(data_save + patient + '/' + patient + 'a_' + trials[j] + '_features_'+str(sec)+'s.npy')))
                                y_train = np.concatenate((y_train, np.load(data_save + patient + '/' + patient + 'a_' + trials[j] + '_labels_'+str(sec)+'s.npy')))
                        else:
                            X_train = np.concatenate((X_train, np.load(data_save + patient + '/' + patient + '_' + trials[j] + '_features_'+str(sec)+'s.npy')))
                            y_train = np.concatenate((y_train, np.load(data_save + patient + '/' + patient + '_' + trials[j] + '_labels_'+str(sec)+'s.npy')))
                if(patient == 'chb17'):
                    if(trials[i+1] == '63'):
                        X_test = np.load(data_save + patient + '/' + patient + 'b_' + trials[i+1] + '_features_'+str(sec)+'s.npy')
                        y_test = np.load(data_save + patient + '/' + patient + 'b_' + trials[i+1] + '_labels_'+str(sec)+'s.npy')
                    else:
                        X_test = np.load(data_save + patient + '/' + patient + 'a_' + trials[i+1] + '_features_'+str(sec)+'s.npy')
                        y_test = np.load(data_save + patient + '/' + patient + 'a_' + trials[i+1] + '_labels_'+str(sec)+'s.npy')
                else:
                    X_test = np.load(data_save + patient + '/' + patient + '_' + trials[i+1] + '_features_'+str(sec)+'s.npy')
                    y_test = np.load(data_save + patient + '/' + patient + '_' + trials[i+1] + '_labels_'+str(sec)+'s.npy')
                # Now we have the training and testing data for this trial.
                # We want to train a xgboost model on this data.
                result = Parallel(n_jobs=n_cores)(delayed(train_and_evaluate_model)(X_train, y_train, random_state,X_test,y_test,weight,complexity,int(sec),'logloss') for weight in list_weight)
                df = pd.DataFrame(result)
                df.to_pickle("results/subject_specific/" + patient + "/" + trials[i+1] + "_xgboost_walk_"+str(sec)+"s.pkl")