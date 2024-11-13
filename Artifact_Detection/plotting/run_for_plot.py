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
import sklearn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
print('The scikit-learn version is {}.'.format(sklearn.__version__))
features = np.load('../data/x_train_250_temporal.npy')
labels = np.load('../data/y_train_binary_250_temporal.npy')
X_train, X_test,y_train, y_test = train_test_split(features, labels, random_state=42)
kb_array = []
accuracy_array = []
ccp_alpha_array = np.linspace(0.000001,0.001,1000)
ccp_alpha_array = np.append(0,ccp_alpha_array)
for ccp in ccp_alpha_array:
    n_estimators = 16
    clf = ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.8, min_samples_leaf=1, min_samples_split=11, n_estimators=n_estimators,n_jobs=-1,ccp_alpha = ccp)
    clf.fit(X_train,y_train)
    results = clf.predict(X_test)
    sum_est = 0
    mem = 0
    total_byte = 9
    for i in range(n_estimators):
        estimator = clf.estimators_[i]
        sum_est += estimator.tree_.node_count
        n_nodes = estimator.tree_.node_count
        mem += n_nodes*total_byte*0.001
    kb = mem + total_byte*n_estimators*0.001
    kb_array.append(kb)
    accuracy_array.append(accuracy_score(y_test,results))
np.save('results/kb_array_v2.npy',kb_array)
np.save('results/accuracy_array_v2.npy',accuracy_array)
features = np.load('../data/x_train_250_temporal_unrolled.npy')
labels = np.load('../data/y_train_multi_binary_250_temporal_unrolled.npy')
X_train, X_test,y_train, y_test = train_test_split(features, labels, random_state=42)
kb_array = []
accuracy_array = []
ccp_alpha_array = np.linspace(0.000001,0.001,1000)
ccp_alpha_array = np.append(0,ccp_alpha_array)
for ccp in ccp_alpha_array:
    n_estimators = 16
    clf = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.35000000000000003, min_samples_leaf=1, min_samples_split=4, n_estimators=n_estimators,n_jobs=-1,ccp_alpha=ccp)
    clf.fit(X_train,y_train)
    results = clf.predict(X_test)
    sum_est = 0
    mem = 0
    total_byte = 9
    for i in range(n_estimators):
        estimator = clf.estimators_[i]
        sum_est += estimator.tree_.node_count
        n_nodes = estimator.tree_.node_count
        mem += n_nodes*total_byte*0.001
    kb = mem + total_byte*n_estimators*0.001
    kb_array.append(kb)
    accuracy_array.append(accuracy_score(y_test,results))
np.save('results/kb_array_MC.npy',kb_array)
np.save('results/accuracy_array_MC.npy',accuracy_array)
features = np.load('../data/x_train_250_temporal_unrolled.npy')
labels = np.load('../data/y_train_multioutput_250_temporal_unrolled.npy')
X_train, X_test,y_train, y_test = train_test_split(features, labels, random_state=42)
n_estimators = 40
kb_array = []
accuracy_array = []
for ccp in ccp_alpha_array:
    n_estimators = 16
    clf = ExtraTreesClassifier(bootstrap=False, criterion="entropy", max_features=0.8, min_samples_leaf=3, min_samples_split=6, n_estimators=n_estimators,n_jobs=-1,ccp_alpha=ccp)
    clf.fit(X_train,y_train)
    results = clf.predict(X_test)
    sum_est = 0
    mem = 0
    total_byte = 9
    for i in range(n_estimators):
        estimator = clf.estimators_[i]
        sum_est += estimator.tree_.node_count
        n_nodes = estimator.tree_.node_count
        mem += n_nodes*total_byte*0.001
    kb = mem + total_byte*n_estimators*0.001
    kb_array.append(kb)
    accuracy_array.append(accuracy_score(y_test,results))
np.save('results/kb_array_MMC.npy',kb_array)
np.save('results/accuracy_array_MMC.npy',accuracy_array)