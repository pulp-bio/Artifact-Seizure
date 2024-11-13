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

import os
import sys

import utils.NEDC.nedc_debug_tools as ndt
import utils.NEDC.nedc_edf_tools as net
import utils.NEDC.nedc_file_tools as nft
import utils.NEDC.nedc_mont_tools as nmt


import numpy as np
import pandas as pd
#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# define default argument values
#
DEF_BSIZE = int(10)
DEF_FORMAT_FLOAT = "float"
DEF_FORMAT_SHORT = "short"
DEF_MODE = False



# declare a global debug object so we can use it in functions
#
dbgl = ndt.Dbgl()


def get_data(edf_file,param_file,scale):
    """
    Outputs the data from the TUH artifact dataset.
    Uses the NEDC code that TUH authors provide.

    Outputs the data in an numpy array.
    """
    mnt = nmt.Montage()
    montage = mnt.load(param_file)
    edf = net.Edf()
    ffile = nft.get_fullpath(edf_file)
    (h, isig) = edf.read_edf(ffile, scale, True)
    mnts = nmt.Montage()
    osig = mnts.apply(isig, montage)
    x_data = []
    for key in osig:
        x_data.append(osig[key])
    x_data_array = np.asarray(x_data)
    return x_data_array
def get_labels(rec_df,whole_duration,mode,path,sampling):
    """
    Outputs the labels for the TUH artifact dataset.
    Has three modes of operation.

    1: mode = 'binary' : Binary classification e.g., Artifact :1 Non-artifact:0 
    and if there is an artifact on one of the channels then the whole sample gets 
    classified as an artifact such that the label dataset has the dimension of 
    (N,1) -> where N is the total number of samples we have.

    2: mode = 'multi_binary' : Binary classification e.g., Artifact:1 Non-artifact:0
    and then have it such that if there is an artifact on one of the channels but 
    not the other then it should get classified as such, then the label dataset 
    has the following dimenson: (N,C) where C is the total number of channels.

    3: mode = 'multioutput' : Multioutput classification, e.g., we have several different 
    types of artifacts available that have been labeled, so in that sense we go for a 
    dataset that has channelwise labeling and the correct artifact label for each 
    artifact so that we have a label dataset with dimensions of (N,C)

    Outputs an error if mode is not one of three ['binary','multi_binary','multioutput']
    Outputs the labels in an numpy array.

    """
    if mode not in ['binary', 'multi_binary','multioutput']:
        raise ValueError("mode must be one of the three ['binary','multi_binary','multioutput']")
    electrodes = 22
    all_labels = []
    for elec in range(electrodes):
        if(elec in rec_df['Electrode'].values):
            new_df = rec_df.loc[rec_df['Electrode'].values  == elec]
            labels = []
            total_samples = 0
            for index, row in new_df.iterrows():
                time_start = np.round(row['Time start'])
                time_end = np.round(row['Time end'])
                duration = np.round(time_end - time_start)
                samples = int(sampling * duration)
                total_samples = samples + total_samples
                classification = int(row['Classification'])
                if(mode == 'binary' or mode == 'multi_binary'):
                    if(classification == 6):
                        classification = 0
                    else:
                        classification = 1
                if(mode == 'multioutput'):
                    if(classification == 6):
                        classification = 0
                c = np.asarray([classification] * samples)
                labels = np.append(labels,c)
            if(total_samples != whole_duration):
                classification = 0
                c = np.asarray([classification] * int(whole_duration - total_samples))
                labels = np.append(labels,c)
        else:
            labels = []
            classification = 0
            c = np.asarray([classification] * int(whole_duration))
            labels = np.append(labels,c)
        all_labels.append(labels)
    true_labels = np.asarray(all_labels)
    if(mode == 'binary'):
        end_labels = []
        for i in range(true_labels.shape[1]):
            classi = 0
            for j in range(22):
                if(true_labels[j,i] == 1):
                    classi = 1
            end_labels.append(classi)
        end_labels = np.asarray(end_labels)
        return end_labels
    if(mode == 'multi_binary' or mode == 'multioutput'):
        return true_labels