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
import os
import mne
from utils.feature_processing import dwt_calc, make_labels, fft_power_calc

data_dir = #CHANGEME
data_save = #CHANGEME

patients = os.listdir(data_dir)
patients.sort()
#only read the diretories that start with 'chb'
patients = [p for p in patients if p.startswith('chb')]
patients.pop()
# Read the file which lists the seizure files, it is named RECORDS-WITH-SEIZURES
seizure_files = pd.read_csv(os.path.join(data_dir, 'RECORDS-WITH-SEIZURES'), header=None)
# Seizure files is a list of all the files that have seizures
# Strip chbXX/ from the beginning of the file name
seizure_files = [f[6:] for f in seizure_files[0]]
# Next for each patient only read the summary file which has the name 'chbXX-summary.txt'
summary_files = [os.path.join(data_dir, p, p + '-summary.txt') for p in patients]
# Each summary file represents one patient
# Make a dictionary that has the patient name as key, the value is a dictionary with the seizure file as key and that value has the start and end times of the seizures.
patient_seizure_dict = {}
for i, patient in enumerate(patients):
    seizure_dict = {}
    # Only look at seizure files that correspond to the patient
    summary = pd.read_csv(summary_files[i], sep='\t', header=None)
    # Remove all capitalization from the summary file
    summary = summary[0].str.lower()
    patient_seizure_file = [f for f in seizure_files if f.startswith(patient)]
    
    for seizure_file in patient_seizure_file:
        times = []
        # Find the line that corresponds to the seizure file, it has the string seizure_file in the line
        line = summary[summary.str.contains(seizure_file)].index[0]
        # Find the place where the number of seizures is written, it is 2 lines below the line where the seizure file is
        num_seizures = summary[line + 3]
        # Read the number of seizures, it folows "number of seizures in file: X"
        num_seizures = int(num_seizures.split(': ')[1])
        # Next we want to read the seizure file start and end time.
        # Start time is right below the File name, and end time is below that.
        start_time = summary[line + 1]
        start_time = start_time.split(': ')[1]
        end_time = summary[line + 2]
        end_time = end_time.split(': ')[1]
        # Next we read the start and end seconds of the seizures.
        # The start and end times are in the format "xxxx seconds" we then convert that to samples (256 samples per second)
        # We know there are num_seizures seizures in this file so we loop over them
        for j in range(num_seizures):
            start_seiz = summary[line + 4 + 2*j]
            start_seiz = start_seiz.split(': ')[1]
            # Strip spaces from start_seiz
            start_seiz = start_seiz.strip()
            start_seiz = int(start_seiz.split(' ')[0])
            start_seiz = start_seiz * 256
            end_seiz = summary[line + 4 + (2*j+1)]
            end_seiz = end_seiz.split(': ')[1]
            end_seiz = end_seiz.strip()
            end_seiz = int(end_seiz.split(' ')[0])
            end_seiz = end_seiz * 256
            # We now have the start and end times of the seizures in samples, we add them to the times list
            times.append([start_seiz, end_seiz])
        # We now have the start and end times of the seizures for this patient, we add them to the seizure_dict
        seizure_dict[seizure_file] = times
    # We now add the seizure_dict to the patient_seizure_dict
    patient_seizure_dict[patient] = seizure_dict

#Next I want to simply loop over every patient and load in the data (temporal data) and make the features and labels.

seconds = [1,2,4,8]
# Loop over all the patients
for patient in patient_seizure_dict:
    print("Processing patient: ", patient)
    # Loop over all the seizure files for this patient
    for seizure_file in patient_seizure_dict[patient]:
        # Read the data

        data = mne.io.read_raw_edf(data_dir+patient+'/'+seizure_file,include=['F7-T7','T7-P7','F8-T8','T8-P8'],verbose = 0)
        if(len(data.info.ch_names)> 0):
            raw_data = data.get_data() * 1000000
            if(raw_data.shape[0] == 5):
                raw_data = np.delete(raw_data,-1,0)
            # Loop over all the seizures in this file
            labels = np.zeros(raw_data.shape[1])
            for seizure in patient_seizure_dict[patient][seizure_file]:
                # Make the labels
                labels[seizure[0]:seizure[1]] = 1
            # Make the features
            raw_data = np.transpose(raw_data)
            for sec in seconds:
                fs = 256
                length = int(sec)
                window_length = fs * length
                level = 4
                # Check if the directory data_save+patient exists, if not create it
                if not os.path.exists(data_save+patient):
                    os.makedirs(data_save+patient)
                # Check if the labels already exist, if not create them
                if not os.path.exists(data_save+patient+'/'+seizure_file[:-4]+'_labels_'+str(sec)+'s.npy'):
                    labels_t = make_labels(labels,fs,length)
                    np.save(data_save+patient+'/'+seizure_file[:-4]+'_labels_'+str(sec)+'s.npy',labels_t)
                if not os.path.exists(data_save+patient+'/'+seizure_file[:-4]+'_features_'+str(sec)+'s.npy'):
                    features = dwt_calc(raw_data, window_length, level)
                    # Save the data
                    np.save(data_save+patient+'/'+seizure_file[:-4]+'_features_'+str(sec)+'s.npy',features)
                if not os.path.exists(data_save+patient+'/'+seizure_file[:-4]+'_features_fft_'+str(sec)+'s.npy'):
                    features_fft = fft_power_calc(raw_data, length)
                    # Save the data
                    np.save(data_save+patient+'/'+seizure_file[:-4]+'_features_fft_'+str(sec)+'s.npy',features_fft)


# Next we loop through all the patients and essentially make global features and labels.
# We do this by going into dir_save and reading all the files for each patient and then making the global features and labels for each second.
# We then save the global features and labels for each patient in the dir_save directory.

# Loop over all the patients (make global DWT features)
for patient in patient_seizure_dict:
    print("Processing patient: ", patient)
    # Loop over all the seizure files for this patient
    files = os.listdir(data_save+patient)
    # Loop over all the seconds
    for sec in seconds:
        # Make the labels
        labels = []
        for file in files:
            if file.endswith('labels_'+str(sec)+'s.npy'):
                #check if file does not include the word global
                if not file.startswith('global'):
                    labels.append(np.load(data_save+patient+'/'+file))
        labels = np.concatenate(labels)
        # Make the features
        features = []
        for file in files:
            if file.endswith('features_'+str(sec)+'s.npy'):
                #check if file does not include the word global
                if not file.startswith('global'):
                    features.append(np.load(data_save+patient+'/'+file))
        features = np.concatenate(features)
        # Save the data
        # Check if the directory data_save+patient exists, if not create it
        if not os.path.exists(data_save+patient):
            os.makedirs(data_save+patient)
        np.save(data_save+patient+'/global_features_'+str(sec)+'s.npy',features)
        np.save(data_save+patient+'/global_labels_'+str(sec)+'s.npy',labels)

# Next we loop through all the patients and essentially make global features and labels.
# Loop over all the patients (DWT Global features)
for i,patient in enumerate(patient_seizure_dict):
    print("Processing patient: ", patient)
    if(i == 0):
        features_1s = np.load(data_save+patient+'/global_features_1s.npy')
        labels_1s = np.load(data_save+patient+'/global_labels_1s.npy')
        features_2s = np.load(data_save+patient+'/global_features_2s.npy')
        labels_2s = np.load(data_save+patient+'/global_labels_2s.npy')
        features_4s = np.load(data_save+patient+'/global_features_4s.npy')
        labels_4s = np.load(data_save+patient+'/global_labels_4s.npy')
        features_8s = np.load(data_save+patient+'/global_features_8s.npy')
        labels_8s = np.load(data_save+patient+'/global_labels_8s.npy')
    else:
        features_1s = np.concatenate((features_1s,np.load(data_save+patient+'/global_features_1s.npy')))
        labels_1s = np.concatenate((labels_1s,np.load(data_save+patient+'/global_labels_1s.npy')))
        features_2s = np.concatenate((features_2s,np.load(data_save+patient+'/global_features_2s.npy')))
        labels_2s = np.concatenate((labels_2s,np.load(data_save+patient+'/global_labels_2s.npy')))
        features_4s = np.concatenate((features_4s,np.load(data_save+patient+'/global_features_4s.npy')))
        labels_4s = np.concatenate((labels_4s,np.load(data_save+patient+'/global_labels_4s.npy')))
        features_8s = np.concatenate((features_8s,np.load(data_save+patient+'/global_features_8s.npy')))
        labels_8s = np.concatenate((labels_8s,np.load(data_save+patient+'/global_labels_8s.npy')))
# Save the data
np.save(data_save+'global_features_1s.npy',features_1s)
np.save(data_save+'global_labels_1s.npy',labels_1s)
np.save(data_save+'global_features_2s.npy',features_2s)
np.save(data_save+'global_labels_2s.npy',labels_2s)
np.save(data_save+'global_features_4s.npy',features_4s)
np.save(data_save+'global_labels_4s.npy',labels_4s)
np.save(data_save+'global_features_8s.npy',features_8s)
np.save(data_save+'global_labels_8s.npy',labels_8s)


# Next we loop through all the patients and essentially make global features and labels.
# Loop over all the patients
for i,patient in enumerate(patient_seizure_dict):
    print("Processing patient: ", patient)
    if(i == 0):
        features_1s = np.load(data_save+patient+'/global_features_fft1s.npy')
        features_2s = np.load(data_save+patient+'/global_features_fft2s.npy')
        features_4s = np.load(data_save+patient+'/global_features_fft4s.npy')
        features_8s = np.load(data_save+patient+'/global_features_fft8s.npy')
    else:
        features_1s = np.concatenate((features_1s,np.load(data_save+patient+'/global_features_fft1s.npy')))
        features_2s = np.concatenate((features_2s,np.load(data_save+patient+'/global_features_fft2s.npy')))
        features_4s = np.concatenate((features_4s,np.load(data_save+patient+'/global_features_fft4s.npy')))
        features_8s = np.concatenate((features_8s,np.load(data_save+patient+'/global_features_fft8s.npy')))
# Save the data
np.save(data_save+'global_features_fft_1s.npy',features_1s)
np.save(data_save+'global_features_fft_2s.npy',features_2s)
np.save(data_save+'global_features_fft_4s.npy',features_4s)
np.save(data_save+'global_features_fft_8s.npy',features_8s)

