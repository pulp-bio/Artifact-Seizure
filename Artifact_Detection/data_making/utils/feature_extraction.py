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
import pandas as pd
import numpy as np 
from sklearn.decomposition import PCA
from pywt import wavedec

def fft_power_calc(x_path, length,temporal, window_fs):
    X_train = np.load(x_path)

    fs = window_fs
    window_length = int(length * fs)
    windows = int(np.floor(X_train.shape[1]/(int(length*fs))))
    if(temporal == True):
        X_train = X_train[[1,2,5,6],:]
        for j in range(windows):
            for i in range(4):
                x = X_train[i,int(fs*j*length):int(fs*(length)*(j+1))]
                X = np.fft.fft(x)
                X_pow = np.abs(X) ** 2
                N = x.shape[0]
                sum_of_above_80_sqrt = np.sqrt(np.sum(X_pow[int(80/(fs/N)):(N//2)]))
                features = np.array([sum_of_above_80_sqrt])
                features.shape = (1,1)
                if(i == 0):
                    true_features = features.copy()
                else:
                    true_features = np.append(features,true_features,axis = 0)
            if(j == 0):
                features_final = true_features.copy()
            else:
                features_final = np.append(true_features,features_final,axis = 1)
    else:    
        for j in range(windows):
            for i in range(22):
                x = X_train[i,int(fs*j*length):int(fs*(length)*(j+1))]
                X = np.fft.fft(x)
                X_pow = np.abs(X) ** 2
                N = x.shape[0]
                sum_of_above_80_sqrt = np.sqrt(np.sum(X_pow[int(80/(fs/N)):(N//2)]))
                features = np.array([sum_of_above_80_sqrt])
                features.shape = (1,1)
                if(i == 0):
                    true_features = features.copy()
                else:
                    true_features = np.append(features,true_features,axis = 0)
            if(j == 0):
                features_final = true_features.copy()
            else:
                features_final = np.append(true_features,features_final,axis = 1)
    return features_final

def dwt_calc(x_path, window_length, level, temporal):

    x = np.load(x_path)
    if(temporal == True):
        x = x[[1,2,5,6],:]
    data = np.transpose(x)
    c = data.shape[0]
    PC = data.shape[1]
    windows = int(np.floor(c / window_length))
    wname = "haar"
    channels = 0
    coeffs = wavedec(data[0:window_length, channels], wname, level=level, axis=0)
    cA4, cD4, cD3, cD2, cD1 = coeffs
    features = np.array(
        [
            np.sum(np.power(cD1, 2)),
            np.sum(np.power(cD2, 2)),
            np.sum(np.power(cD3, 2)),
            np.sum(np.power(cD4, 2)),
        ]
    )
    features.shape = (4, 1)
    one = np.ones(shape=1) * features[0]
    two = np.ones(shape=1) * features[1]
    three = np.ones(shape=1) * features[2]
    four = np.ones(shape=1) * features[3]
    total_features = np.array([one, two, three, four])
    true_features = total_features.copy()

    for i in range(1, windows):
        coeffs = wavedec(
            data[(window_length * i) : (window_length * (i + 1)), channels],
            wname,
            level=level,
            axis=0,
        )
        cA4, cD4, cD3, cD2, cD1 = coeffs
        features = np.array(
            [
                np.sum(np.power(cD1, 2)),
                np.sum(np.power(cD2, 2)),
                np.sum(np.power(cD3, 2)),
                np.sum(np.power(cD4, 2)),
            ]
        )
        features.shape = (4, 1)
        one = np.ones(shape=1) * features[0]
        two = np.ones(shape=1) * features[1]
        three = np.ones(shape=1) * features[2]
        four = np.ones(shape=1) * features[3]
        total_features = np.array([one, two, three, four])
        true_features = np.append(true_features, total_features, axis=1)

    features_final = true_features.copy()
    for channels in range(1, PC):
        coeffs = wavedec(data[0:window_length, channels], wname, level=level, axis=0)
        cA4, cD4, cD3, cD2, cD1 = coeffs
        features = np.array(
            [
                np.sum(np.power(cD1, 2)),
                np.sum(np.power(cD2, 2)),
                np.sum(np.power(cD3, 2)),
                np.sum(np.power(cD4, 2)),
            ]
        )
        features.shape = (4, 1)
        one = np.ones(shape=1) * features[0]
        two = np.ones(shape=1) * features[1]
        three = np.ones(shape=1) * features[2]
        four = np.ones(shape=1) * features[3]
        total_features = np.array([one, two, three, four])
        true_features = total_features.copy()

        for i in range(1, windows):
            coeffs = wavedec(
                data[(window_length * i) : (window_length * (i + 1)), channels],
                wname,
                level=level,
                axis=0,
            )
            cA4, cD4, cD3, cD2, cD1 = coeffs
            features = np.array(
                [
                    np.sum(np.power(cD1, 2)),
                    np.sum(np.power(cD2, 2)),
                    np.sum(np.power(cD3, 2)),
                    np.sum(np.power(cD4, 2)),
                ]
            )
            features.shape = (4, 1)
            one = np.ones(shape=1) * features[0]
            two = np.ones(shape=1) * features[1]
            three = np.ones(shape=1) * features[2]
            four = np.ones(shape=1) * features[3]
            total_features = np.array([one, two, three, four])
            true_features = np.append(true_features, total_features, axis=1)
        features_final = np.append(features_final, true_features, axis=0)

    features_final = np.transpose(features_final)
    return features_final

def make_labels(y_path, mode, temporal,fs,length):
    y = np.load(y_path)
    #length = 1
    if(temporal == False):
        if(mode == 'binary'):
            windows = int(np.floor(y.shape[0]/(int(length*fs))))
            true_labels = []
            for i in range(windows):
                sum_of_labels = np.sum(y[i*fs*length:(i+1)*fs*length])
                if(sum_of_labels > fs/2):
                    true_labels.append(1)
                else:
                    true_labels.append(0)
            all_labels = np.array(true_labels)

        elif(mode == 'multi_binary'):
            windows = int(np.floor(y.shape[1]/(int(length*fs))))
            for channel in range(22):
                true_labels = []
                for i in range(windows):
                    sum_of_labels = np.sum(y[channel,i*fs*length:(i+1)*fs*length])
                    if(sum_of_labels > fs/2):
                        true_labels.append(1)
                    else:
                        true_labels.append(0)
                true_labels = np.array(true_labels)
                true_labels.shape = (true_labels.shape[0],1)
                if(channel == 0):
                    all_labels = true_labels.copy()
                else:
                    all_labels = np.append(all_labels,true_labels,axis=1)
        elif(mode == 'multioutput'):
            y = y.astype('int64')
            windows = int(np.floor(y.shape[1]/(int(length*fs))))
            for channel in range(22):
                true_labels = []
                for i in range(0, windows):
                    chosen_label = np.bincount(y[channel,i*fs*length:(i+1)*fs*length]).argmax()
                    true_labels.append(chosen_label)
                true_labels = np.array(true_labels)
                true_labels.shape = (true_labels.shape[0],1)
                if(channel == 0):
                    all_labels = true_labels.copy()
                else:
                    all_labels = np.append(all_labels,true_labels,axis=1)

    else:
        if(mode == 'binary'):
            #Here we need the multibinary path for this to work since we need to redo the way the labels were made.
            end_labels = []
            temp_channels = [1,2,5,6]
            for i in range(y.shape[1]):
                classi = 0
                for j in temp_channels:
                    if(y[j,i] == 1):
                        classi = 1
                end_labels.append(classi)
            end_labels = np.asarray(end_labels)
            windows = int(np.floor(end_labels.shape[0]/(int(length*fs))))
            true_labels = []
            for i in range(windows):
                sum_of_labels = np.sum(end_labels[i*fs*length:(i+1)*fs*length])
                if(sum_of_labels > fs/2):
                    true_labels.append(1)
                else:
                    true_labels.append(0)
            all_labels = np.array(true_labels)
        elif(mode == 'multi_binary'):
            temp_channels = [1,2,5,6]
            windows = int(np.floor(y.shape[1]/(int(length*fs))))
            counter = 0
            for channel in temp_channels:
                true_labels = []
                for i in range(windows):
                    sum_of_labels = np.sum(y[channel,i*fs*length:(i+1)*fs*length])
                    if(sum_of_labels > fs/2):
                        true_labels.append(1)
                    else:
                        true_labels.append(0)
                true_labels = np.array(true_labels)
                true_labels.shape = (true_labels.shape[0],1)
                if(counter == 0):
                    all_labels = true_labels.copy()
                    counter = counter + 1
                else:
                    all_labels = np.append(all_labels,true_labels,axis=1)
                    counter = counter + 1
        elif(mode == 'multioutput'):
            temp_channels = [1,2,5,6]
            y = y.astype('int64')
            windows = int(np.floor(y.shape[1]/(int(length*fs))))
            counter = 0
            for channel in temp_channels:
                true_labels = []
                for i in range(0, windows):
                    chosen_label = np.bincount(y[channel,i*fs*length:(i+1)*fs*length]).argmax()
                    true_labels.append(chosen_label)
                true_labels = np.array(true_labels)
                true_labels.shape = (true_labels.shape[0],1)
                if(counter == 0):
                    all_labels = true_labels.copy()
                else:
                    all_labels= np.append(all_labels,true_labels,axis=1)
                counter = counter + 1
    return all_labels