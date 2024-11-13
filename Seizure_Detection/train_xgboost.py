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
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import time
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import os
def train_test_model(X, y, seed,X_test,y_test,weight , complexity):
    model = XGBClassifier(n_estimators=complexity, random_state=seed, scale_pos_weight = weight, max_depth = 15, eval_metric = 'logloss',use_label_encoder=False)
    model.fit(X, y)
    y_pred = model.predict(X_test)
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    for i in range(len(y_pred)):
        pred = y_pred[i]
        true = y_test[i]
        if pred == 1 and true == 1:
            TP = TP + 1
        elif pred == 1 and true == 0:
            FP = FP + 1
        elif pred == 0 and true == 0:
            TN = TN + 1
        else:
            FN = FN + 1
    if(TP + FN != 0):
        sensitivity = float((TP / (TP + FN)) * 100)
    else:
        sensitivity = float(0)
    if(TN + FP != 0):
        specificity = float((TN / (TN + FP)) * 100)
    else:
        specificity = float(0)

    accuracy = (TP + TN) / (TN + FN + TP + FP) * 100
    if(TP + FP != 0):    
        precision = float((TP / (TP + FP)) * 100)
    else:
        precision = float(0)
    try:
        f1_score = (2 * ((precision*sensitivity)/(precision+sensitivity)))
    except ZeroDivisionError:
        f1_score = float(0)
    return accuracy,sensitivity,specificity,precision,f1_score,complexity,weight


results_dir = 'results/global/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
print("Training on features without overlap")
random_state = 0
data_save = #CHANGEME
seconds = 1
features_final = np.load(data_save+'global_features_'+str(seconds)+'s.npy')
true_labels = np.load(data_save+'global_labels_'+str(seconds)+'s.npy')
X_train, X_test, y_train, y_test = train_test_split(
    features_final, true_labels, test_size=0.25, random_state=random_state
)
n_cores = 2
list_weight = [i for i in range(1,1000)]
complexity = 100

print("Start training 1s")
start = time.time()
result = Parallel(n_jobs=n_cores)(delayed(train_test_model)(X_train, y_train, random_state,X_test,y_test,weight,complexity) for weight in list_weight)
end = time.time()
df = pd.DataFrame(result)
df.to_pickle("results/global/xgboost_global_1s_depth_15.pkl")
print('done took: ',end - start, 's')

seconds = 2
features_final = np.load(data_save+'global_features_'+str(seconds)+'s.npy')
true_labels = np.load(data_save+'global_labels_'+str(seconds)+'s.npy')
X_train, X_test, y_train, y_test = train_test_split(
    features_final, true_labels, test_size=0.25, random_state=random_state
)
print("Start training 2s")
start = time.time()
result = Parallel(n_jobs=n_cores)(delayed(train_test_model)(X_train, y_train, random_state,X_test,y_test,weight,complexity) for weight in list_weight)
end = time.time()
df = pd.DataFrame(result)
df.to_pickle("results/global/xgboost_global_2s_depth_15.pkl")
print('done took: ',end - start, 's')

seconds = 4
features_final = np.load(data_save+'global_features_'+str(seconds)+'s.npy')
true_labels = np.load(data_save+'global_labels_'+str(seconds)+'s.npy')
X_train, X_test, y_train, y_test = train_test_split(
    features_final, true_labels, test_size=0.25, random_state=random_state
)
print("Start training 4s")
start = time.time()
result = Parallel(n_jobs=n_cores)(delayed(train_test_model)(X_train, y_train, random_state,X_test,y_test,weight,complexity) for weight in list_weight)
end = time.time()
df = pd.DataFrame(result)
df.to_pickle("results/global/xgboost_global_4s_depth_15.pkl")
print('done took: ',end - start, 's')
seconds = 8
features_final = np.load(data_save+'global_features_'+str(seconds)+'s.npy')
true_labels = np.load(data_save+'global_labels_'+str(seconds)+'s.npy')
X_train, X_test, y_train, y_test = train_test_split(
    features_final, true_labels, test_size=0.25, random_state=random_state
)
print("Start training 8s")
start = time.time()
result = Parallel(n_jobs=n_cores)(delayed(train_test_model)(X_train, y_train, random_state,X_test,y_test,weight,complexity) for weight in list_weight)
end = time.time()
df = pd.DataFrame(result)
df.to_pickle("results/global/xgboost_global_8s_depth_15.pkl")
print('done took: ',end - start, 's')