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
from xgboost import XGBClassifier
def metric_calc(y_pred, y_true):
    """
    Calculate binary classification metrics: accuracy, sensitivity, specificity,
    precision, and F1-score, based on true and predicted values.

    Parameters:
    y_pred (list): Predicted binary labels
    y_true (list): True binary labels

    Returns:
    tuple: Contains accuracy, sensitivity, specificity, precision, and F1-score
    """
    # Initialize counts for True Positives, False Negatives, False Positives, and True Negatives
    TP, FN, FP, TN = 0, 0, 0, 0

    # Count occurrences of each classification outcome
    for pred, true in zip(y_pred, y_true):
        if pred == 1 and true == 1:
            TP += 1
        elif pred == 1 and true == 0:
            FP += 1
        elif pred == 0 and true == 0:
            TN += 1
        else:
            FN += 1

    # Calculate metrics with zero checks to prevent division by zero
    sensitivity = ((TP / (TP + FN)) * 100) if (TP + FN) else 0
    specificity = ((TN / (TN + FP)) * 100) if (TN + FP) else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN) * 100
    precision = ((TP / (TP + FP)) * 100) if (TP + FP) else 0
    f1_score = (2 * (precision * sensitivity) / (precision + sensitivity)) if (precision + sensitivity) else 0

    return accuracy, sensitivity, specificity, precision, f1_score

def sliding_window_majority_backwards(y_pred):
    """
    Apply a sliding window with a backwards majority rule to smooth predictions.
    Uses the current and two previous values for majority voting.

    Parameters:
    y_pred (list): Binary prediction sequence

    Returns:
    list: Smoothed binary prediction sequence
    """
    y_pred_new = y_pred.copy()
    for i in range(len(y_pred)):
        if(i == 0):
            sums = 0 + 0 + y_pred[i]
        elif(i == 1):
            sums = 0 + y_pred[i-1] + y_pred[i]
        else:
            sums = y_pred[i-2] + y_pred[i-1] + y_pred[i]
        if(sums > 1):
            y_pred_new[i] = 1
        else:
            y_pred_new[i] = 0
    return y_pred_new

def sliding_window_majority_middle(y_pred):
    """
    Apply a sliding window with a centered majority rule to smooth predictions.
    Uses the current, previous, and next values for majority voting.

    Parameters:
    y_pred (list): Binary prediction sequence

    Returns:
    list: Smoothed binary prediction sequence
    """
    y_pred_new = y_pred.copy()
    for i in range(len(y_pred)):
        if(i == 0):
            sums = 0 + y_pred[i] + y_pred[i+1]
        elif(i == len(y_pred)-1):
            sums = y_pred[i-1] + y_pred[i] + 0
        else:
            sums = y_pred[i-1] + y_pred[i] + y_pred[i+1]
        if(sums > 1):
            y_pred_new[i] = 1
        else:
            y_pred_new[i] = 0
    return y_pred_new

def sliding_window_majority_middle_extended(y_pred):
    """
    Apply an extended sliding window with a centered majority rule to smooth predictions.
    Uses the current, two previous, and two next values for majority voting.

    Parameters:
    y_pred (list): Binary prediction sequence

    Returns:
    list: Smoothed binary prediction sequence
    """
    y_pred_new = y_pred.copy()
    for i in range(len(y_pred)):
        if(i == 0):
            sums = 0 + 0 + y_pred[i] + y_pred[i+1] + y_pred[i+2]
        elif(i == 1):
            sums = 0 + y_pred[i-1] + y_pred[i] + y_pred[i+1] + y_pred[i+2]
        elif(i == len(y_pred)-2):
            sums = y_pred[i-2] + y_pred[i-1] + y_pred[i] + y_pred[i+1] + 0
        elif(i == len(y_pred)-1):
            sums = y_pred[i-2] + y_pred[i-1] + y_pred[i] + 0 + 0
        else:
            sums = y_pred[i-2] + y_pred[i-1] + y_pred[i] + y_pred[i+1] + y_pred[i+2]
        if(sums > 2):
            y_pred_new[i] = 1
        else:
            y_pred_new[i] = 0
    return y_pred_new

def drop_anomaly(predicted, observed, interval):
    """
    Drop false positive predictions over a specified interval based on observed data.
    Useful for filtering anomalies from the prediction sequence.

    Parameters:
    predicted (list): Binary prediction sequence
    observed (list): Binary observation sequence for comparison
    interval (int): Interval length in seconds

    Returns:
    list: Modified binary prediction sequence with anomalies dropped
    """
    span = int(60 * 15 / interval)
    mod_pred = predicted.copy()
    for idx in range(len(observed) - 1):
        if observed[idx] == 1 and observed[idx + 1] == 0:
            span = min(span, len(observed) - idx)
            for offset in range(span):
                mod_pred[idx + offset] = 0
    return mod_pred


def train_and_evaluate_model(X, y, seed, X_test, y_test, weight, complexity, sec, eval_metric='logloss'):
    """
    Train an XGBoost model on the training data and evaluate it on the test data.
    This function applies anomaly filtering and smoothing on predictions, and 
    calculates performance metrics on raw and smoothed predictions.

    Parameters:
    X (ndarray): Training feature matrix.
    y (ndarray): Training labels.
    seed (int): Random seed for reproducibility.
    X_test (ndarray): Test feature matrix.
    y_test (ndarray): Test labels.
    weight (float): Weight to balance positive and negative classes.
    complexity (int): Number of trees (complexity) in the model.
    sec (int): Interval parameter for anomaly removal.
    eval_metric (str): Evaluation metric for XGBoost training (default is 'logloss').

    Returns:
    tuple: Sensitivity and specificity for both raw and smoothed predictions, 
           along with model complexity and weight.
    """
    # Initialize and configure the XGBoost model
    model = XGBClassifier(
        n_estimators=complexity,      # Set number of trees based on model complexity
        random_state=seed,            # Seed for reproducibility
        scale_pos_weight=weight,      # Class balancing weight
        objective='binary:logistic',  # Binary classification objective
        eval_metric=eval_metric       # Evaluation metric
    )

    # Train the model on training data
    model.fit(X, y)
    
    # Generate predictions on test data
    y_pred = model.predict(X_test)

    # Apply anomaly filtering to predictions
    y_pred_filtered = drop_anomaly(y_pred, y_test, sec)

    # Apply sliding window smoothing to the filtered predictions
    y_pred_smoothed = sliding_window_majority_middle(y_pred_filtered)

    # Calculate metrics on filtered predictions
    _, sensitivity, specificity, _, _ = metric_calc(y_pred, y_test)

    # Calculate metrics on smoothed predictions
    _, sensitivity_s, specificity_s, _, _ = metric_calc(y_pred_smoothed, y_test)

    # Return relevant metrics and model parameters
    return sensitivity, specificity, sensitivity_s, specificity_s, complexity, weight

