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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from joblib import Parallel, delayed
import time

def train_model(training_features,training_target,testing_features,testing_target, ccp):
    """
    Train a model using the given features and labels and the given parameters
    :param features: The features to train the model on
    :param labels: The labels to train the model on
    :param ccp: The ccp_alpha parameter to use
    :return: 
    """

    clf = GradientBoostingClassifier(max_features=0.35000000000000003, min_samples_leaf=1, min_samples_split=4, n_estimators=32, max_depth=19,ccp_alpha=ccp)
    clf.fit(training_features, training_target)
    results = clf.predict(testing_features)
    sum_est = 0
    mem = 0
    total_byte = 8
    n_estimators=32
    for i in range(n_estimators):
        estimator = clf.estimators_[i][0]
        sum_est += estimator.tree_.node_count
        n_nodes = estimator.tree_.node_count
        mem += n_nodes*total_byte*0.001
    kb = mem + total_byte*n_estimators*0.001
    acc = accuracy_score(testing_target, results)
    
    return acc, kb, ccp

features = np.load('../data/x_train_250_temporal_unrolled.npy')
labels = np.load('../data/y_train_multi_binary_250_temporal_unrolled.npy')
X_train, X_test,y_train, y_test = train_test_split(features, labels, random_state=42)
start = 1/(1000*1000*1000*1000*1000)
end = 1/(1000)
n_points = 2500
ccp_alpha_array = np.linspace(start,end,n_points)

print("Start training gradient")
start = time.time()
result = Parallel(n_jobs=-1)(delayed(train_model)(X_train,y_train,X_test,y_test,ccp) for ccp in ccp_alpha_array)
end = time.time()
df = pd.DataFrame(result)
df.to_pickle("results/gradient_results_mmc_32.pkl")
print('done took: ',end - start, 's')