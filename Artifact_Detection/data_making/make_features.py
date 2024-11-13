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
import time
from utils.feature_extraction import dwt_calc,fft_power_calc, make_labels
import argparse
from argparse import RawTextHelpFormatter

def main():
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--frequencies', default = "250", help = 'Which set of frequencies to make features for, must be in class [250, 250_1000, 256, 256_512, all].')
    args = parser.parse_args()
    SAVE_PATH = 'CHANGETHISTOYOURPATH'

    if(args.frequencies == '250'):
        print("250Hz selected")
        print("Running DWT power calculations")
        x_path = SAVE_PATH + '/X_train_250.npy'
        temporal = False
        window_length = 250
        level = 4
        start = time.time()
        features_DWT = dwt_calc(x_path, window_length, level,temporal)
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        print("Running FFT power calculations")
        start = time.time()
        features_FFT = fft_power_calc(x_path,1,temporal,250)
        features_FFT = np.transpose(features_FFT)
        features_final = np.append(features_DWT, features_FFT,axis=1)
        
        np.save('../data/x_train_250.npy',features_final)
        del features_final, features_DWT, features_FFT
        print("Running DWT power calculations temporal")
        temporal = True
        window_length = 250
        level = 4
        start = time.time()
        features_DWT = dwt_calc(x_path, window_length, level,temporal)
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        print("Running FFT power calculations temporal")
        start = time.time()
        features_FFT = fft_power_calc(x_path,1,temporal,250)
        features_FFT = np.transpose(features_FFT)
        features_final = np.append(features_DWT, features_FFT,axis=1)
        
        np.save('../data/x_train_250_temporal.npy',features_final)
        del features_final, features_DWT, features_FFT
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        y_path = SAVE_PATH + '/y_train_binary_250.npy'
        temporal = False
        mode = 'binary'
        fs = 250
        print("Generating labels for binary 250Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_binary_250.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multi_binary'
        y_path = SAVE_PATH + '/y_train_multi_binary_250.npy'
        fs = 250
        print("Generating labels for multi_binary 250Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multi_binary_250.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multioutput'
        y_path = SAVE_PATH + '/y_train_multioutput_250.npy'
        fs = 250
        print("Generating labels for multioutput 250Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multioutput_250.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )


        y_path = SAVE_PATH + '/y_train_multi_binary_250.npy'
        temporal = True
        mode = 'binary'
        fs = 250
        print("Generating labels for binary 250Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_binary_250_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multi_binary'
        y_path = SAVE_PATH + '/y_train_multi_binary_250.npy'
        fs = 250
        print("Generating labels for multi_binary 250Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multi_binary_250_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multioutput'
        y_path = SAVE_PATH + '/y_train_multioutput_250.npy'
        fs = 250
        print("Generating labels for multioutput 250Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multioutput_250_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

    elif(args.frequencies == '250_1000'):
        print("250Hz and 1000Hz selected")
        print("Running DWT power calculations")
        x_path = SAVE_PATH + '/X_train_250_1000.npy'
        temporal = False
        window_length = 250
        level = 4
        start = time.time()
        features_DWT = dwt_calc(x_path, window_length, level,temporal)
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        print("Running FFT power calculations")
        start = time.time()
        features_FFT = fft_power_calc(x_path,1,temporal,250)
        features_FFT = np.transpose(features_FFT)
        features_final = np.append(features_DWT, features_FFT,axis=1)
        
        np.save('../data/x_train_250_1000.npy',features_final)
        del features_final, features_DWT, features_FFT
        print("Running DWT power calculations temporal")
        temporal = True
        window_length = 250
        level = 4
        start = time.time()
        features_DWT = dwt_calc(x_path, window_length, level,temporal)
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        print("Running FFT power calculations temporal")
        start = time.time()
        features_FFT = fft_power_calc(x_path,1,temporal,250)
        features_FFT = np.transpose(features_FFT)
        features_final = np.append(features_DWT, features_FFT,axis=1)
        
        np.save('../data/x_train_250_1000_temporal.npy',features_final)
        del features_final, features_DWT, features_FFT
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        y_path = SAVE_PATH + '/y_train_binary_250_1000.npy'
        mode = 'binary'
        fs = 250
        temporal = False
        print("Generating labels for binary 250Hz and 1000Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_binary_250_1000.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multi_binary'
        y_path = SAVE_PATH + '/y_train_multi_binary_250_1000.npy'
        fs = 250
        print("Generating labels for multi_binary 250Hz and 1000Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multi_binary_250_1000.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multioutput'
        y_path = SAVE_PATH + '/y_train_multioutput_250_1000.npy'
        fs = 250
        print("Generating labels for multioutput 250Hz and 1000Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multioutput_250_1000.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )


        y_path = SAVE_PATH + '/y_train_multi_binary_250_1000.npy'
        temporal = True
        mode = 'binary'
        fs = 250
        print("Generating labels for binary 250Hz and 1000Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_binary_250_1000_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multi_binary'
        y_path = SAVE_PATH + '/y_train_multi_binary_250_1000.npy'
        fs = 250
        print("Generating labels for multi_binary 250Hz and 1000Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multi_binary_250_1000_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multioutput'
        y_path = SAVE_PATH + '/y_train_multioutput_250_1000.npy'
        fs = 250
        print("Generating labels for multioutput 250Hz and 1000Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multioutput_250_1000_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )


    elif(args.frequencies == '256'):
        print("256Hz selected")
        print("Running DWT power calculations")
        x_path = SAVE_PATH + '/X_train_256.npy'
        temporal = False
        window_length = 256
        level = 4
        start = time.time()
        features_DWT = dwt_calc(x_path, window_length, level,temporal)
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        print("Running FFT power calculations")
        start = time.time()
        features_FFT = fft_power_calc(x_path,1,temporal,256)
        features_FFT = np.transpose(features_FFT)
        features_final = np.append(features_DWT, features_FFT,axis=1)
        
        np.save('../data/x_train_256.npy',features_final)
        del features_final, features_DWT, features_FFT
        print("Running DWT power calculations temporal")
        temporal = True
        window_length = 256
        level = 4
        start = time.time()
        features_DWT = dwt_calc(x_path, window_length, level,temporal)
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        print("Running FFT power calculations temporal")
        start = time.time()
        features_FFT = fft_power_calc(x_path,1,temporal,256)
        features_FFT = np.transpose(features_FFT)
        features_final = np.append(features_DWT, features_FFT,axis=1)
        
        np.save('../data/x_train_256_temporal.npy',features_final)
        del features_final, features_DWT, features_FFT
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        y_path = SAVE_PATH + '/y_train_binary_256.npy'
        mode = 'binary'
        fs = 256
        temporal = False
        print("Generating labels for binary 256Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_binary_256.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multi_binary'
        y_path = SAVE_PATH + '/y_train_multi_binary_256.npy'
        fs = 256
        print("Generating labels for multi_binary 256Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multi_binary_256.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multioutput'
        y_path = SAVE_PATH + '/y_train_multioutput_256.npy'
        fs = 256
        print("Generating labels for multioutput 250Hz and 1000Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multioutput_256.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )


        y_path = SAVE_PATH + '/y_train_multi_binary_256.npy'
        temporal = True
        mode = 'binary'
        fs = 256
        print("Generating labels for binary 256Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_binary_256_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multi_binary'
        y_path = SAVE_PATH + '/y_train_multi_binary_256.npy'
        fs = 256
        print("Generating labels for multi_binary 256Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multi_binary_256_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multioutput'
        y_path = SAVE_PATH + '/y_train_multioutput_256.npy'
        fs = 256
        print("Generating labels for multioutput 256Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multioutput_256_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

    elif(args.frequencies == '256_512'):
        print("256Hz and 512Hz selected")
        print("Running DWT power calculations")
        x_path = SAVE_PATH + '/X_train_256_512.npy'
        temporal = False
        window_length = 256
        level = 4
        start = time.time()
        features_DWT = dwt_calc(x_path, window_length, level,temporal)
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        print("Running FFT power calculations")
        start = time.time()
        features_FFT = fft_power_calc(x_path,1,temporal,256)
        features_FFT = np.transpose(features_FFT)
        features_final = np.append(features_DWT, features_FFT,axis=1)
        
        np.save('../data/x_train_256_512.npy',features_final)
        del features_final, features_DWT, features_FFT
        print("Running DWT power calculations temporal")
        temporal = True
        window_length = 256
        level = 4
        start = time.time()
        features_DWT = dwt_calc(x_path, window_length, level,temporal)
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        print("Running FFT power calculations temporal")
        start = time.time()
        features_FFT = fft_power_calc(x_path,1,temporal,256)
        features_FFT = np.transpose(features_FFT)
        features_final = np.append(features_DWT, features_FFT,axis=1)
        
        np.save('../data/x_train_256_512_temporal.npy',features_final)
        del features_final, features_DWT, features_FFT
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        y_path = SAVE_PATH + '/y_train_binary_256_512.npy'
        mode = 'binary'
        fs = 256
        temporal = False
        print("Generating labels for binary 256Hz and 512Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_binary_256_512.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multi_binary'
        y_path = SAVE_PATH + '/y_train_multi_binary_256_512.npy'
        fs = 256
        print("Generating labels for multi_binary 256Hz and 512Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multi_binary_256_512.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multioutput'
        y_path = SAVE_PATH + '/y_train_multioutput_256_512.npy'
        fs = 256
        print("Generating labels for multioutput 256Hz and 512Hz")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multioutput_256_512.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )


        y_path = SAVE_PATH + '/y_train_multi_binary_256_512.npy'
        temporal = True
        mode = 'binary'
        fs = 256
        print("Generating labels for binary 256Hz and 512Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_binary_256_512_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multi_binary'
        y_path = SAVE_PATH + '/y_train_multi_binary_256_512.npy'
        fs = 256
        print("Generating labels for multi_binary 256Hz and 512Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multi_binary_256_512_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multioutput'
        y_path = SAVE_PATH + '/y_train_multioutput_256_512.npy'
        fs = 256
        print("Generating labels for multioutput 256Hz and 512Hz temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multioutput_256_512_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )
    
    elif(args.frequencies == 'all'):
        print("all selected")
        print("Running DWT power calculations")
        x_path = SAVE_PATH + '/X_train_all.npy'
        temporal = False
        window_length = 250
        level = 4
        start = time.time()
        features_DWT = dwt_calc(x_path, window_length, level,temporal)
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        print("Running FFT power calculations")
        start = time.time()
        features_FFT = fft_power_calc(x_path,1,temporal,250)
        features_FFT = np.transpose(features_FFT)
        features_final = np.append(features_DWT, features_FFT,axis=1)
        
        np.save('../data/x_train_all.npy',features_final)
        del features_final, features_DWT, features_FFT
        print("Running DWT power calculations temporal")
        temporal = True
        window_length = 250
        level = 4
        start = time.time()
        features_DWT = dwt_calc(x_path, window_length, level,temporal)
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        print("Running FFT power calculations temporal")
        start = time.time()
        features_FFT = fft_power_calc(x_path,1,temporal,250)
        features_FFT = np.transpose(features_FFT)
        features_final = np.append(features_DWT, features_FFT,axis=1)
        
        np.save('../data/x_train_all_temporal.npy',features_final)
        del features_final, features_DWT, features_FFT
        end = time.time()
        print("Done took: " + str(end - start) + " seconds" )
        y_path = SAVE_PATH + '/y_train_binary_all.npy'
        mode = 'binary'
        fs = 250
        temporal = False
        print("Generating labels for binary all")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_binary_all.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multi_binary'
        y_path = SAVE_PATH + '/y_train_multi_binary_all.npy'
        fs = 250
        print("Generating labels for multi_binary all")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multi_binary_all.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multioutput'
        y_path = SAVE_PATH + '/y_train_multioutput_all.npy'
        fs = 250
        print("Generating labels for multioutput all")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multioutput_all.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )


        y_path = SAVE_PATH + '/y_train_multi_binary_all.npy'
        temporal = True
        mode = 'binary'
        fs = 250
        print("Generating labels for binary all temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_binary_all_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multi_binary'
        y_path = SAVE_PATH + '/y_train_multi_binary_all.npy'
        fs = 250
        print("Generating labels for multi_binary all temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multi_binary_all_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

        mode = 'multioutput'
        y_path = SAVE_PATH + '/y_train_multioutput_all.npy'
        fs = 250
        print("Generating labels for multioutput all temporal")
        start = time.time()
        true_labels = make_labels(y_path, mode, temporal,fs)
        end = time.time()
        np.save('../data/y_train_multioutput_all_temporal.npy',true_labels)
        print("Done took: " + str(end - start) + " seconds" )

if __name__ == '__main__':
    main()