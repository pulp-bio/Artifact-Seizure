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

import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tpot import TPOTClassifier
import time
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import argparse
import sys
from argparse import RawTextHelpFormatter
from sklearn.preprocessing import StandardScaler


def main():
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--frequencies', default = "250", help = 'Which set of features to test, must be in class [250,250_1000,256,256_512,all].')
    parser.add_argument('--mode', default = 'BI', help= 'What mode to train on, must be in class [BI, MB, MO]')
    parser.add_argument('--generations',type=int, default = 5, help = 'Number of iterations to the run pipeline optimization process. It must be a positive number or None.')
    parser.add_argument('--pop_size', type=int,default = 50, help = 'Number of individuals to retain in the genetic programming population every generation. Must be a positive number.')
    parser.add_argument('--CV', type=int,default = 5, help = 'integer, to specify the number of folds in a StratifiedKFold.')
    parser.add_argument('--unroll', default='Yes', help='If to use unrolled data or not, must be in [Yes, No]')
    args = parser.parse_args()
    
    if(args.unroll == 'Yes'):
        print("Running search for unrolled data")
        if(args.frequencies == '250'):
            print("Choosing 250Hz frequency to train on")
            if(args.mode == 'BI'):
                print("Choosing Binary labels to train on")
                features = np.load('data/x_train_250.npy')
                features_temporal = np.load('data/x_train_250_temporal.npy')
                transformer_standard = StandardScaler().fit(features_temporal)
                features_temporal = transformer_standard.transform(features_temporal)
                labels = np.load('data/y_train_binary_250.npy')
                labels_temporal = np.load('data/y_train_binary_250_temporal.npy')
            elif(args.mode == 'MB'):
                print("Choosing MultiBinary labels to train on")
                features = np.load('data/x_train_250_unrolled.npy')
                features_temporal = np.load('data/x_train_250_temporal_unrolled.npy')
                labels = np.load('data/y_train_multi_binary_250_unrolled.npy')
                labels_temporal = np.load('data/y_train_multi_binary_250_temporal_unrolled.npy')
            elif(args.mode == 'MO'):
                print("Choosing MultiOutput labels to train on")
                features = np.load('data/x_train_250_unrolled.npy')
                features_temporal = np.load('data/x_train_250_temporal_unrolled.npy')
                labels = np.load('data/y_train_multioutput_250_unrolled.npy')
                labels_temporal = np.load('data/y_train_multioutput_250_temporal_unrolled.npy')
            else:
                print("No valid set of mode selected, please choose from [BI,MB,MO]")
                print("Exiting training ..........")
                sys.exit()
        elif(args.frequencies == '250_1000'):
            print("Choosing 250Hz + 1000Hz frequency to train on")
            if(args.mode == 'BI'):
                print("Choosing Binary labels to train on")
                features = np.load('data/x_train_250_1000.npy')
                features_temporal = np.load('data/x_train_250_1000_temporal.npy')
                labels = np.load('data/y_train_binary_250_1000.npy')
                labels_temporal = np.load('data/y_train_binary_250_1000_temporal.npy')
            elif(args.mode == 'MB'):
                print("Choosing MultiBinary labels to train on")
                features = np.load('data/x_train_250_1000_unrolled.npy')
                features_temporal = np.load('data/x_train_250_1000_temporal_unrolled.npy')
                labels = np.load('data/y_train_multi_binary_250_1000_unrolled.npy')
                labels_temporal = np.load('data/y_train_multi_binary_250_1000_temporal_unrolled.npy')
            elif(args.mode == 'MO'):
                print("Choosing MultiOutput labels to train on")
                features = np.load('data/x_train_250_1000_unrolled.npy')
                features_temporal = np.load('data/x_train_250_1000_temporal_unrolled.npy')
                labels = np.load('data/y_train_multioutput_250_1000_unrolled.npy')
                labels_temporal = np.load('data/y_train_multioutput_250_1000_temporal_unrolled.npy')
            else:
                print("No valid set of mode selected, please choose from [BI,MB,MO]")
                print("Exiting training ..........")
                sys.exit()
        elif(args.frequencies == '256'):
            print("Choosing 256Hz frequency to train on")
            if(args.mode == 'BI'):
                print("Choosing Binary labels to train on")
                features = np.load('data/x_train_256.npy')
                features_temporal = np.load('data/x_train_256_temporal.npy')
                labels = np.load('data/y_train_binary_256.npy')
                labels_temporal = np.load('data/y_train_binary_256_temporal.npy')
            elif(args.mode == 'MB'):
                print("Choosing MultiBinary labels to train on")
                features = np.load('data/x_train_256_unrolled.npy')
                features_temporal = np.load('data/x_train_256_temporal_unrolled.npy')
                labels = np.load('data/y_train_multi_binary_256_unrolled.npy')
                labels_temporal = np.load('data/y_train_multi_binary_256_temporal_unrolled.npy')
            elif(args.mode == 'MO'):
                print("Choosing MultiOutput labels to train on")
                features = np.load('data/x_train_256_unrolled.npy')
                features_temporal = np.load('data/x_train_256_temporal_unrolled.npy')
                labels = np.load('data/y_train_multioutput_256_unrolled.npy')
                labels_temporal = np.load('data/y_train_multioutput_256_temporal_unrolled.npy')
            else:
                print("No valid set of mode selected, please choose from [BI,MB,MO]")
                print("Exiting training ..........")
        elif(args.frequencies == '256_512'):
            print("Choosing 256Hz + 512Hz frequency to train on")
            if(args.mode == 'BI'):
                print("Choosing Binary labels to train on")
                features = np.load('data/x_train_256_512.npy')
                features_temporal = np.load('data/x_train_256_512_temporal.npy')
                labels = np.load('data/y_train_binary_256_512.npy')
                labels_temporal = np.load('data/y_train_binary_256_512_temporal.npy')
            elif(args.mode == 'MB'):
                print("Choosing MultiBinary labels to train on")
                features = np.load('data/x_train_256_512_unrolled.npy')
                features_temporal = np.load('data/x_train_256_512_temporal_unrolled.npy')
                labels = np.load('data/y_train_multi_binary_256_512_unrolled.npy')
                labels_temporal = np.load('data/y_train_multi_binary_256_512_temporal_unrolled.npy')
            elif(args.mode == 'MO'):
                print("Choosing MultiOutput labels to train on")
                features = np.load('data/x_train_256_512_unrolled.npy')
                features_temporal = np.load('data/x_train_256_512_temporal_unrolled.npy')
                labels = np.load('data/y_train_multioutput_256_512_unrolled.npy')
                labels_temporal = np.load('data/y_train_multioutput_256_512_temporal_unrolled.npy')
            else:
                print("No valid set of mode selected, please choose from [BI,MB,MO]")
                print("Exiting training ..........")
        elif(args.frequencies == 'all'):
            print("Choosing all frequencies to train on")
            if(args.mode == 'BI'):
                print("Choosing Binary labels to train on")
                features = np.load('data/x_train_all.npy')
                features_temporal = np.load('data/x_train_all_temporal.npy')
                labels = np.load('data/y_train_binary_all.npy')
                labels_temporal = np.load('data/y_train_binary_all_temporal.npy')
            elif(args.mode == 'MB'):
                print("Choosing MultiBinary labels to train on")
                features = np.load('data/x_train_all_unrolled.npy')
                features_temporal = np.load('data/x_train_all_temporal_unrolled.npy')
                labels = np.load('data/y_train_multi_binary_all_unrolled.npy')
                labels_temporal = np.load('data/y_train_multi_binary_all_temporal_unrolled.npy')
            elif(args.mode == 'MO'):
                print("Choosing MultiOutput labels to train on")
                features = np.load('data/x_train_all_unrolled.npy')
                features_temporal = np.load('data/x_train_all_temporal_unrolled.npy')
                labels = np.load('data/y_train_multioutput_all_unrolled.npy')
                labels_temporal = np.load('data/y_train_multioutput_all_temporal_unrolled.npy')
            else:
                print("No valid set of mode selected, please choose from [BI,MB,MO]")
                print("Exiting training ..........")
        else:
            print("No valid set of frequencies selected, please choose from [250,250_1000,256,256_512,all]")
            print("Exiting training ..........")
            sys.exit()
    else:
        print("Running search for normal data")
        if(args.frequencies == '250'):
            print("Choosing 250Hz frequency to train on")
            if(args.mode == 'BI'):
                print("Choosing Binary labels to train on")
                features = np.load('data/x_train_250.npy')
                features_temporal = np.load('data/x_train_250_temporal.npy')
                labels = np.load('data/y_train_binary_250.npy')
                labels_temporal = np.load('data/y_train_binary_250_temporal.npy')
            elif(args.mode == 'MB'):
                print("Choosing MultiBinary labels to train on")
                features = np.load('data/x_train_250.npy')
                features_temporal = np.load('data/x_train_250_temporal.npy')
                labels = np.load('data/y_train_multi_binary_250.npy')
                labels_temporal = np.load('data/y_train_multi_binary_250_temporal.npy')
            elif(args.mode == 'MO'):
                print("Choosing MultiOutput labels to train on")
                features = np.load('data/x_train_250.npy')
                features_temporal = np.load('data/x_train_250_temporal.npy')
                labels = np.load('data/y_train_multioutput_250.npy')
                labels_temporal = np.load('data/y_train_multioutput_250_temporal.npy')
            else:
                print("No valid set of mode selected, please choose from [BI,MB,MO]")
                print("Exiting training ..........")
                sys.exit()
        elif(args.frequencies == '250_1000'):
            print("Choosing 250Hz + 1000Hz frequency to train on")
            if(args.mode == 'BI'):
                print("Choosing Binary labels to train on")
                features = np.load('data/x_train_250_1000.npy')
                features_temporal = np.load('data/x_train_250_1000_temporal.npy')
                labels = np.load('data/y_train_binary_250_1000.npy')
                labels_temporal = np.load('data/y_train_binary_250_1000_temporal.npy')
            elif(args.mode == 'MB'):
                print("Choosing MultiBinary labels to train on")
                features = np.load('data/x_train_250_1000.npy')
                features_temporal = np.load('data/x_train_250_1000_temporal.npy')
                labels = np.load('data/y_train_multi_binary_250_1000.npy')
                labels_temporal = np.load('data/y_train_multi_binary_250_1000_temporal.npy')
            elif(args.mode == 'MO'):
                print("Choosing MultiOutput labels to train on")
                features = np.load('data/x_train_250_1000.npy')
                features_temporal = np.load('data/x_train_250_1000_temporalnpy')
                labels = np.load('data/y_train_multioutput_250_1000.npy')
                labels_temporal = np.load('data/y_train_multioutput_250_1000_temporal.npy')
            else:
                print("No valid set of mode selected, please choose from [BI,MB,MO]")
                print("Exiting training ..........")
                sys.exit()
        elif(args.frequencies == '256'):
            print("Choosing 256Hz frequency to train on")
            if(args.mode == 'BI'):
                print("Choosing Binary labels to train on")
                features = np.load('data/x_train_256.npy')
                features_temporal = np.load('data/x_train_256_temporal.npy')
                labels = np.load('data/y_train_binary_256.npy')
                labels_temporal = np.load('data/y_train_binary_256_temporal.npy')
            elif(args.mode == 'MB'):
                print("Choosing MultiBinary labels to train on")
                features = np.load('data/x_train_256.npy')
                features_temporal = np.load('data/x_train_256_temporal.npy')
                labels = np.load('data/y_train_multi_binary_256.npy')
                labels_temporal = np.load('data/y_train_multi_binary_256_temporal.npy')
            elif(args.mode == 'MO'):
                print("Choosing MultiOutput labels to train on")
                features = np.load('data/x_train_256.npy')
                features_temporal = np.load('data/x_train_256_temporal.npy')
                labels = np.load('data/y_train_multioutput_256.npy')
                labels_temporal = np.load('data/y_train_multioutput_256_temporal.npy')
            else:
                print("No valid set of mode selected, please choose from [BI,MB,MO]")
                print("Exiting training ..........")
        elif(args.frequencies == '256_512'):
            print("Choosing 256Hz + 512Hz frequency to train on")
            if(args.mode == 'BI'):
                print("Choosing Binary labels to train on")
                features = np.load('data/x_train_256_512.npy')
                features_temporal = np.load('data/x_train_256_512_temporal.npy')
                labels = np.load('data/y_train_binary_256_512.npy')
                labels_temporal = np.load('data/y_train_binary_256_512_temporal.npy')
            elif(args.mode == 'MB'):
                print("Choosing MultiBinary labels to train on")
                features = np.load('data/x_train_256_512.npy')
                features_temporal = np.load('data/x_train_256_512_temporal.npy')
                labels = np.load('data/y_train_multi_binary_256_512.npy')
                labels_temporal = np.load('data/y_train_multi_binary_256_512_temporal.npy')
            elif(args.mode == 'MO'):
                print("Choosing MultiOutput labels to train on")
                features = np.load('data/x_train_256_512.npy')
                features_temporal = np.load('data/x_train_256_512_temporal.npy')
                labels = np.load('data/y_train_multioutput_256_512.npy')
                labels_temporal = np.load('data/y_train_multioutput_256_512_temporal.npy')
            else:
                print("No valid set of mode selected, please choose from [BI,MB,MO]")
                print("Exiting training ..........")
        elif(args.frequencies == 'all'):
            print("Choosing all frequencies to train on")
            if(args.mode == 'BI'):
                print("Choosing Binary labels to train on")
                features = np.load('data/x_train_all.npy')
                features_temporal = np.load('data/x_train_all_temporal.npy')
                labels = np.load('data/y_train_binary_all.npy')
                labels_temporal = np.load('data/y_train_binary_all_temporal.npy')
            elif(args.mode == 'MB'):
                print("Choosing MultiBinary labels to train on")
                features = np.load('data/x_train_all.npy')
                features_temporal = np.load('data/x_train_all_temporal.npy')
                labels = np.load('data/y_train_multi_binary_all.npy')
                labels_temporal = np.load('data/y_train_multi_binary_all_temporal.npy')
            elif(args.mode == 'MO'):
                print("Choosing MultiOutput labels to train on")
                features = np.load('data/x_train_all.npy')
                features_temporal = np.load('data/x_train_all_temporal.npy')
                labels = np.load('data/y_train_multioutput_all.npy')
                labels_temporal = np.load('data/y_train_multioutput_all_temporal.npy')
            else:
                print("No valid set of mode selected, please choose from [BI,MB,MO]")
                print("Exiting training ..........")
        else:
            print("No valid set of frequencies selected, please choose from [250,250_1000,256,256_512,all]")
            print("Exiting training ..........")
            sys.exit()

    SEED = 42
    train_this = False
    if(train_this):
        X, X_test, y, y_test = train_test_split(
            features, labels, test_size=0.15, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.176, random_state=42)

        tpot = TPOTClassifier(
            generations=args.generations,
            population_size=args.pop_size,
            random_state=SEED,
            n_jobs=-1, # cuML requires n_jobs=1, the default
            cv=args.CV,
            verbosity=2,
            scoring='accuracy'
        )
        print("Starting to train Accuracy scoring")
        start = time.time()
        tpot.fit(X_train, y_train)

        y_pred = tpot.predict(X_val)
        accuracy = accuracy_score(y_pred,y_val)
        f = open("exported_pipelines/metrics_"+str(args.frequencies)+"_"+str(args.mode)+"_"+str(args.unroll)+".txt", "a")
        f.write("Accuracy, Scoring\n")
        f.write(str(accuracy)+", Accuracy\n")
        f.close()
        tpot.predict(X_test)
        tpot.export('exported_pipelines/exported_pipeline_'+str(args.frequencies)+"_"+str(args.mode)+"_"+str(args.unroll)+'_'+str(args.generations)+'_'+str(args.pop_size)+'_'+str(args.CV)+'.py')
        end = time.time()
        print("All done took: " + str(end - start) + 's')
        print("Test score: "+str(accuracy_score(y_pred,y_val)))

    X, X_test, y, y_test = train_test_split(
        features_temporal, labels_temporal, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.176, random_state=42)

    tpot = TPOTClassifier(
        generations=args.generations,
        population_size=args.pop_size,
        random_state=SEED,
        n_jobs=-1, # cuML requires n_jobs=1, the default
        cv=args.CV,
        verbosity=2,
        scoring='accuracy'
    )
    print("Starting to train Accuracy scoring (Temporal)")
    start = time.time()
    tpot.fit(X_train, y_train)

    y_pred = tpot.predict(X_val)
    accuracy = accuracy_score(y_pred,y_val)
    f = open("exported_pipelines/metrics_"+str(args.frequencies)+"_"+str(args.mode)+"_"+str(args.unroll)+"_temporal.txt", "a")
    f.write("Accuracy, Scoring\n")
    f.write(str(accuracy)+", Accuracy\n")
    f.close()
    tpot.predict(X_test)
    tpot.export('exported_pipelines/exported_pipeline_'+str(args.frequencies)+"_"+str(args.mode)+"_"+str(args.unroll)+'_'+str(args.generations)+'_'+str(args.pop_size)+'_'+str(args.CV)+'_temporal.py')
    end = time.time()
    print("All done took: " + str(end - start) + 's')
    print("Test score: "+str(accuracy_score(y_pred,y_val)))

if __name__ == '__main__':
    main()



