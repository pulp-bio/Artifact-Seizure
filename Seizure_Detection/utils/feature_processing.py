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

from pywt import wavedec
import numpy as np
def dwt_calc(data, window_length, level):

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

def make_labels(labels,fs,length):
    windows = int(np.floor(labels.shape[0]/(int(length*fs))))
    true_labels = []
    for i in range(windows):
        sum_of_labels = np.sum(labels[i*fs*length:(i+1)*fs*length])
        if(sum_of_labels > fs/2):
            true_labels.append(1)
        else:
            true_labels.append(0)
    all_labels = np.array(true_labels)
    return all_labels


def fft_power_calc(X_train, length):

    fs = 256
    windows = int(np.floor(X_train.shape[0]/(int(length*fs))))
    for j in range(windows):
        for i in range(4):
            x = X_train[int(fs*j*length):int(fs*(length)*(j+1)),i]
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
    features_final = np.transpose(features_final)
    return features_final