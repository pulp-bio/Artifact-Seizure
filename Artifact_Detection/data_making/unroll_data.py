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

def unroll_labels(data, output_file, temporal=False):
    end_labels = None
    if(temporal):
        ch = 4
    else:
        ch = 22
    for channel in range(ch):
        temp = data[:, channel]
        if channel == 0:
            end_labels = temp.copy()
        else:
            end_labels = np.append(end_labels, temp, axis=0)
    np.save(output_file, end_labels)
    del end_labels, data

def unroll_features(data, output_file, temporal=False):
    end_features = None
    if(temporal):
        ch = 4
    else:
        ch = 22
    for i in range(ch):
        temp = data[:, i*5:(i+1)*5]
        if i == 0:
            end_features = temp.copy()
        else:
            end_features = np.append(end_features, temp, axis=0)
    np.save(output_file, end_features)
    del end_features, data

def process_labels(datasets_list, temporal = False):
    if temporal:
        ch = 4
    else:
        ch = 22
    for dataset in datasets_list:
        for i in range(ch):
            for j in range(dataset.shape[0]):
                if dataset[j, i] == 21:
                    dataset[j, i] = 1
                elif dataset[j, i] == 22:
                    dataset[j, i] = 2
                elif dataset[j, i] == 23:
                    dataset[j, i] = 3
                elif dataset[j, i] == 24:
                    dataset[j, i] = 4
                elif dataset[j, i] == 30:
                    dataset[j, i] = 5
                elif dataset[j, i] == 100:
                    dataset[j, i] = 6
                elif dataset[j, i] == 101:
                    dataset[j, i] = 7
                elif dataset[j, i] == 102:
                    dataset[j, i] = 8
                elif dataset[j, i] == 103:
                    dataset[j, i] = 9
                elif dataset[j, i] == 105:
                    dataset[j, i] = 10
                elif dataset[j, i] == 106:
                    dataset[j, i] = 11
                elif dataset[j, i] == 109:
                    dataset[j, i] = 12

# Load and unroll y_train_multi_binary data
y_train_multi_binary_250 = np.load('../data/y_train_multi_binary_250.npy')
unroll_labels(y_train_multi_binary_250, '../data/y_train_multi_binary_250_unrolled.npy')

y_train_multi_binary_250_1000 = np.load('../data/y_train_multi_binary_250_1000.npy')
unroll_labels(y_train_multi_binary_250_1000, '../data/y_train_multi_binary_250_1000_unrolled.npy')

y_train_multi_binary_256 = np.load('../data/y_train_multi_binary_256.npy')
unroll_labels(y_train_multi_binary_256, '../data/y_train_multi_binary_256_unrolled.npy')

y_train_multi_binary_256_512 = np.load('../data/y_train_multi_binary_256_512.npy')
unroll_labels(y_train_multi_binary_256_512, '../data/y_train_multi_binary_256_512_unrolled.npy')

y_train_multi_binary_all = np.load('../data/y_train_multi_binary_all.npy')
unroll_labels(y_train_multi_binary_all, '../data/y_train_multi_binary_all_unrolled.npy')

# Load and unroll y_train_multi_binary_temporal data
y_train_multi_binary_250_temporal = np.load('../data/y_train_multi_binary_250_temporal.npy')
unroll_labels(y_train_multi_binary_250_temporal, '../data/y_train_multi_binary_250_temporal_unrolled.npy', True)

y_train_multi_binary_250_1000_temporal = np.load('../data/y_train_multi_binary_250_1000_temporal.npy')
unroll_labels(y_train_multi_binary_250_1000_temporal, '../data/y_train_multi_binary_250_1000_temporal_unrolled.npy', True)

y_train_multi_binary_256_temporal = np.load('../data/y_train_multi_binary_256_temporal.npy')
unroll_labels(y_train_multi_binary_256_temporal, '../data/y_train_multi_binary_256_temporal_unrolled.npy', True)

y_train_multi_binary_256_512_temporal = np.load('../data/y_train_multi_binary_256_512_temporal.npy')
unroll_labels(y_train_multi_binary_256_512_temporal, '../data/y_train_multi_binary_256_512_temporal_unrolled.npy', True)

y_train_multi_binary_all_temporal = np.load('../data/y_train_multi_binary_all_temporal.npy')
unroll_labels(y_train_multi_binary_all_temporal, '../data/y_train_multi_binary_all_temporal_unrolled.npy', True)

# Load and unroll x_train data
x_train_250 = np.load('../data/x_train_250.npy')
unroll_features(x_train_250, '../data/x_train_250_unrolled.npy')

x_train_250_1000 = np.load('../data/x_train_250_1000.npy')
unroll_features(x_train_250_1000, '../data/x_train_250_1000_unrolled.npy')

x_train_256 = np.load('../data/x_train_256.npy')
unroll_features(x_train_256, '../data/x_train_256_unrolled.npy')

x_train_256_512 = np.load('../data/x_train_256_512.npy')
unroll_features(x_train_256_512, '../data/x_train_256_512_unrolled.npy')

x_train_all = np.load('../data/x_train_all.npy')
unroll_features(x_train_all, '../data/x_train_all_unrolled.npy')

# Load and unroll x_train_temporal data
x_train_250_temporal = np.load('../data/x_train_250_temporal.npy')
unroll_features(x_train_250_temporal, '../data/x_train_250_temporal_unrolled.npy', True)

x_train_250_1000_temporal = np.load('../data/x_train_250_1000_temporal.npy')
unroll_features(x_train_250_1000_temporal, '../data/x_train_250_1000_temporal_unrolled.npy', True)

x_train_256_temporal = np.load('../data/x_train_256_temporal.npy')
unroll_features(x_train_256_temporal, '../data/x_train_256_temporal_unrolled.npy', True)

x_train_256_512_temporal = np.load('../data/x_train_256_512_temporal.npy')
unroll_features(x_train_256_512_temporal, '../data/x_train_256_512_temporal_unrolled.npy', True)

x_train_all_temporal = np.load('../data/x_train_all_temporal.npy')
unroll_features(x_train_all_temporal, '../data/x_train_all_temporal_unrolled.npy', True)

# Load and process y_train_multioutput data
y_train_multioutput_250 = np.load('../data/y_train_multioutput_250.npy')
y_train_multioutput_250_1000 = np.load('../data/y_train_multioutput_250_1000.npy')
y_train_multioutput_256 = np.load('../data/y_train_multioutput_256.npy')
y_train_multioutput_256_512 = np.load('../data/y_train_multioutput_256_512.npy')
y_train_multioutput_all = np.load('../data/y_train_multioutput_all.npy')

datasets_list = [y_train_multioutput_250, y_train_multioutput_250_1000, y_train_multioutput_256, y_train_multioutput_256_512, y_train_multioutput_all]
process_labels(datasets_list)

np.save('../data/y_train_multioutput_250.npy', y_train_multioutput_250)
np.save('../data/y_train_multioutput_250_1000.npy', y_train_multioutput_250_1000)
np.save('../data/y_train_multioutput_256.npy', y_train_multioutput_256)
np.save('../data/y_train_multioutput_256_512.npy', y_train_multioutput_256_512)
np.save('../data/y_train_multioutput_all.npy', y_train_multioutput_all)

# Load and process y_train_multioutput_temporal data
y_train_multioutput_250_temporal = np.load('../data/y_train_multioutput_250_temporal.npy')
y_train_multioutput_250_1000_temporal = np.load('../data/y_train_multioutput_250_1000_temporal.npy')
y_train_multioutput_256_temporal = np.load('../data/y_train_multioutput_256_temporal.npy')
y_train_multioutput_256_512_temporal = np.load('../data/y_train_multioutput_256_512_temporal.npy')
y_train_multioutput_all_temporal = np.load('../data/y_train_multioutput_all_temporal.npy')

datasets_list = [y_train_multioutput_250_temporal, y_train_multioutput_250_1000_temporal, y_train_multioutput_256_temporal, y_train_multioutput_256_512_temporal, y_train_multioutput_all_temporal]
process_labels(datasets_list, True)

np.save('../data/y_train_multioutput_250_temporal.npy', y_train_multioutput_250_temporal)
np.save('../data/y_train_multioutput_250_1000_temporal.npy', y_train_multioutput_250_1000_temporal)
np.save('../data/y_train_multioutput_256_temporal.npy', y_train_multioutput_256_temporal)
np.save('../data/y_train_multioutput_256_512_temporal.npy', y_train_multioutput_256_512_temporal)
np.save('../data/y_train_multioutput_all_temporal.npy', y_train_multioutput_all_temporal)
