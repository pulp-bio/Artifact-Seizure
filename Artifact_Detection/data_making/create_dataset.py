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
from utils.data_loading import get_data,get_labels
import numpy as np
import pandas as pd
import mne
import argparse
import sys
from argparse import RawTextHelpFormatter

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--frequencies', default = "250", help = 'Which set of frequencies to make dataset for, must be in class [250, 250_1000, 256, 256_512, all].')
    parser.add_argument('--mode', default = "binary", help = 'Which mode to calculate dataset for, must be in class [binary, multi_binary, multioutput].')
    PATH_CHANGE = 'CHANGETHISTOYOURPATH'
    SAVE_PATH = 'CHANGETHISTOYOURPATH'
    edf_files = pd.read_csv(PATH_CHANGE+'/TUH/artifacts/lists/edf_01_tcp_ar.list',header=None)
    rec_files = pd.read_csv(PATH_CHANGE+'/TUH/artifacts/lists/rec_01_tcp_ar.list',header=None)
    args = parser.parse_args()
    #mode = args.mode
    total_length = []
    skip_file = []
    counter = 0
    scale = False
    output_fs = 250
    if(args.frequencies == '250'):
        print("Making data for 250Hz")
        #Make dataset for 250Hz
        for index,row in edf_files.iterrows():
            path = PATH_CHANGE + 'TUH/artifacts/' + row[0][3:]
            raw = mne.io.read_raw_edf(path,verbose=0)
            x_data_array = get_data(path,'utils/test_params.txt',scale)
            if(raw.info['sfreq']==250):
                skip_file.append(1)
                total_length.append(x_data_array.shape[1])
                if(counter%10 == 0 and counter!=0):
                    print("Proccessed " + str(counter) + " number of data arrays")
                if(counter == 0):
                    X_train = x_data_array.copy()
                else:
                    X_train = np.append(X_train,x_data_array,axis=1)
                counter += 1
            else:
                skip_file.append(0)
    elif(args.frequencies == '250_1000'):
        print("Making data for 250Hz and 1000Hz")
        #Make dataset for 250 and 1000Hz
        for index,row in edf_files.iterrows():
            path = PATH_CHANGE + 'TUH/artifacts/' + row[0][3:]
            raw = mne.io.read_raw_edf(path,verbose=0)
            x_data_array = get_data(path,'utils/test_params.txt',scale)
            if(raw.info['sfreq']==250):
                skip_file.append(1)
                total_length.append(x_data_array.shape[1])
                if(counter%10 == 0 and counter!=0):
                    print("Proccessed " + str(counter) + " number of data arrays")
                if(counter == 0):
                    X_train = x_data_array.copy()
                else:
                    X_train = np.append(X_train,x_data_array,axis=1)
                counter += 1
            elif(raw.info['sfreq'] == 1000):
                skip_file.append(1)
                new_scale = output_fs / 1000
                # calculate new length of sample
                new_length = round(x_data_array.shape[1] * new_scale)
                new_x_data_array = x_data_array[:,::4]
                total_length.append(new_length)
                if(counter%10 == 0 and counter!=0):
                    print("Proccessed " + str(counter) + " number of data arrays")
                if(counter == 0):
                    X_train = new_x_data_array.copy()
                else:
                    X_train = np.append(X_train,new_x_data_array,axis=1)
                counter += 1
            else:
                skip_file.append(0)
    elif(args.frequencies == '256'):
        print("Making data for 256Hz")
        #Make dataset for 256Hz
        for index,row in edf_files.iterrows():
            path = PATH_CHANGE+'/TUH/artifacts/' + row[0][3:]
            raw = mne.io.read_raw_edf(path,verbose=0)
            x_data_array = get_data(path,'utils/test_params.txt',scale)
            if(raw.info['sfreq']==256):
                skip_file.append(2)
                total_length.append(x_data_array.shape[1])
                if(counter%10 == 0 and counter!=0):
                    print("Proccessed " + str(counter) + " number of data arrays")
                if(counter == 0):
                    X_train = x_data_array.copy()
                else:
                    X_train = np.append(X_train,x_data_array,axis=1)
                counter += 1
            else:
                skip_file.append(0)
    elif(args.frequencies == '256_512'):
        print("Making data for 256Hz and 512Hz")
        #Make dataset for 256 and 512Hz
        for index,row in edf_files.iterrows():
            path = PATH_CHANGE + '/TUH/artifacts/' + row[0][3:]
            raw = mne.io.read_raw_edf(path,verbose=0)
            x_data_array = get_data(path,'utils/test_params.txt',scale)
            if(raw.info['sfreq']==256):
                skip_file.append(2)
                total_length.append(x_data_array.shape[1])
                if(counter%10 == 0 and counter!=0):
                    print("Proccessed " + str(counter) + " number of data arrays")
                if(counter == 0):
                    X_train = x_data_array.copy()
                else:
                    X_train = np.append(X_train,x_data_array,axis=1)
                counter += 1
            elif(raw.info['sfreq'] == 512):
                skip_file.append(2)
                new_scale = 256 / 512
                # calculate new length of sample
                new_length = round(x_data_array.shape[1] * new_scale)
                new_x_data_array = x_data_array[:,::2]
                total_length.append(new_length)
                if(counter%10 == 0 and counter!=0):
                    print("Proccessed " + str(counter) + " number of data arrays")
                if(counter == 0):
                    X_train = new_x_data_array.copy()
                else:
                    X_train = np.append(X_train,new_x_data_array,axis=1)
                counter += 1
            else:
                skip_file.append(0)

    elif(args.frequencies == 'all'):
        print("Making data for all frequencies")
        #Make dataset for all frequencies (this resamples)
        for index,row in edf_files.iterrows():  
            path = PATH_CHANGE + '/TUH/artifacts/' + row[0][3:]
            raw = mne.io.read_raw_edf(path,verbose=0)
            x_data_array = get_data(path,'utils/test_params.txt',scale)
            if(raw.info['sfreq']==250):
                skip_file.append(1)
                total_length.append(x_data_array.shape[1])
                if(counter%10 == 0 and counter!=0):
                    print("Proccessed " + str(counter) + " number of data arrays")
                if(counter == 0):
                    X_train = x_data_array.copy()
                else:
                    X_train = np.append(X_train,x_data_array,axis=1)
                counter += 1
            else:
                skip_file.append(1)
                input_fs = raw.info['sfreq']
                new_scale = output_fs / input_fs
                # calculate new length of sample
                new_length = round(x_data_array.shape[1] * new_scale)
                new_x_data_array = np.zeros(shape=(22,new_length),dtype='float64')
                total_length.append(new_length)
                for channels in range(22):
                    new_x_data_array[channels,:] = np.interp(
                                                        np.linspace(0.0, 1.0, new_length, endpoint=False),  # where to interpret
                                                        np.linspace(0.0, 1.0, x_data_array.shape[1], endpoint=False),  # known positions
                                                        x_data_array[channels,:],  # known data points
                                                        )
                if(counter%10 == 0 and counter!=0):
                    print("Proccessed " + str(counter) + " number of data arrays")
                if(counter == 0):
                    X_train = new_x_data_array.copy()
                else:
                    X_train = np.append(X_train,new_x_data_array,axis=1)
                counter += 1
    total_length = np.asarray(total_length)
    print("Went over a total of " + str(counter) + " data-arrays")
    np.save(SAVE_PATH+'/X_train_' + str(args.frequencies) + '.npy',X_train)
    del X_train
    #Make Binary labels
    counter = 0
    mode = 'binary'
    for index,row in rec_files.iterrows():
        if(skip_file[index] == 0):
            continue
        elif(skip_file[index] == 1):
            sampling = 250
            path = PATH_CHANGE + '/TUH/artifacts/' + row[0][3:]
            rec_df = pd.read_csv(path, header=None,names=['Electrode','Time start', 'Time end', 'Classification'])
            labels = get_labels(rec_df,total_length[counter],mode,path,sampling)
            if(counter%10 == 0 and counter!=0):
                print("Proccessed " + str(counter) + " number of label arrays")
            if(counter == 0):
                y_train = labels.copy()
            else:
                if(mode == 'binary'):
                    y_train = np.append(y_train,labels)
                else:
                    y_train = np.append(y_train,labels,axis=1)
            counter += 1
        elif(skip_file[index] == 2):
            sampling = 256
            path = PATH_CHANGE + '/TUH/artifacts/' + row[0][3:]
            rec_df = pd.read_csv(path, header=None,names=['Electrode','Time start', 'Time end', 'Classification'])
            labels = get_labels(rec_df,total_length[counter],mode,path,sampling)
            if(counter%10 == 0 and counter!=0):
                print("Proccessed " + str(counter) + " number of label arrays")
            if(counter == 0):
                y_train = labels.copy()
            else:
                if(mode == 'binary'):
                    y_train = np.append(y_train,labels)
                else:
                    y_train = np.append(y_train,labels,axis=1)
            counter += 1
    print("Went over a total of " + str(counter) + " labels")
    np.save(SAVE_PATH+'/y_train_' + str(mode) + '_' + str(args.frequencies) + '.npy',y_train)

    #Make MultiBinary labels
    counter = 0
    mode = 'multi_binary'
    for index,row in rec_files.iterrows():
        if(skip_file[index] == 0):
            continue
        elif(skip_file[index] == 1):
            sampling = 250
            path = PATH_CHANGE + '/TUH/artifacts/' + row[0][3:]
            rec_df = pd.read_csv(path, header=None,names=['Electrode','Time start', 'Time end', 'Classification'])
            labels = get_labels(rec_df,total_length[counter],mode,path,sampling)
            if(counter%10 == 0 and counter!=0):
                print("Proccessed " + str(counter) + " number of label arrays")
            if(counter == 0):
                y_train = labels.copy()
            else:
                if(mode == 'binary'):
                    y_train = np.append(y_train,labels)
                else:
                    y_train = np.append(y_train,labels,axis=1)
            counter += 1
        elif(skip_file[index] == 2):
            sampling = 256
            path = PATH_CHANGE + '/TUH/artifacts/' + row[0][3:]
            rec_df = pd.read_csv(path, header=None,names=['Electrode','Time start', 'Time end', 'Classification'])
            labels = get_labels(rec_df,total_length[counter],mode,path,sampling)
            if(counter%10 == 0 and counter!=0):
                print("Proccessed " + str(counter) + " number of label arrays")
            if(counter == 0):
                y_train = labels.copy()
            else:
                if(mode == 'binary'):
                    y_train = np.append(y_train,labels)
                else:
                    y_train = np.append(y_train,labels,axis=1)
            counter += 1
    print("Went over a total of " + str(counter) + " labels")
    np.save(SAVE_PATH+'/y_train_' + str(mode) + '_' + str(args.frequencies) + '.npy',y_train)

    #Make MultioutputLabels
    counter = 0
    mode = 'multioutput'
    for index,row in rec_files.iterrows():
        if(skip_file[index] == 0):
            continue
        elif(skip_file[index]==1):
            sampling = 250
            path = PATH_CHANGE + '/TUH/artifacts/' + row[0][3:]
            rec_df = pd.read_csv(path, header=None,names=['Electrode','Time start', 'Time end', 'Classification'])
            labels = get_labels(rec_df,total_length[counter],mode,path,sampling)
            if(counter%10 == 0 and counter!=0):
                print("Proccessed " + str(counter) + " number of label arrays")
            if(counter == 0):
                y_train = labels.copy()
            else:
                if(mode == 'binary'):
                    y_train = np.append(y_train,labels)
                else:
                    y_train = np.append(y_train,labels,axis=1)
            counter += 1
        elif(skip_file[index] == 2):
            sampling = 256
            path = PATH_CHANGE + '/TUH/artifacts/' + row[0][3:]
            rec_df = pd.read_csv(path, header=None,names=['Electrode','Time start', 'Time end', 'Classification'])
            labels = get_labels(rec_df,total_length[counter],mode,path,sampling)
            if(counter%10 == 0 and counter!=0):
                print("Proccessed " + str(counter) + " number of label arrays")
            if(counter == 0):
                y_train = labels.copy()
            else:
                if(mode == 'binary'):
                    y_train = np.append(y_train,labels)
                else:
                    y_train = np.append(y_train,labels,axis=1)
            counter += 1
    print("Went over a total of " + str(counter) + " labels")
    np.save(SAVE_PATH + '/y_train_' + str(mode) + '_' + str(args.frequencies) + '.npy',y_train)

if __name__ == '__main__':
    main()