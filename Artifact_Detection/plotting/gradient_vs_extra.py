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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


# Load the data
df = pd.read_pickle("results/gradient_results_32.pkl")
x_array_v2 = np.load('results/kb_array_v2.npy')
y_array_v2 = np.load('results/accuracy_array_v2.npy')
x_array = df[1].to_numpy()
y_array = df[0].to_numpy() 
df_refined_1 = pd.read_pickle("results/gradient_results_32_fine.pkl")
df_refined_2 = pd.read_pickle("results/gradient_results_32_finer.pkl")
x_array_refined_1 = df_refined_1[1].to_numpy()
y_array_refined_1 = df_refined_1[0].to_numpy()
x_array_refined_2 = df_refined_2[1].to_numpy()
y_array_refined_2 = df_refined_2[0].to_numpy()
y_array_refined_1 = moving_average(y_array_refined_1, 50)
y_array_refined_2 = moving_average(y_array_refined_2, 50)
x_array_refined_1 = x_array_refined_1[50:]
x_array_refined_2 = x_array_refined_2[50:]
x_array = np.concatenate((x_array,x_array_refined_1,x_array_refined_2))
y_array = np.concatenate((y_array,y_array_refined_1,y_array_refined_2))
sort = np.argsort(x_array)
x_array = x_array[sort]
y_array = y_array[sort]


#Find the increase in accuracy at L1 capacity and and L2 capacity for Mr.Wolf
# We say L1 capacity is 40 for mr wolf beacuse the L1 has a size of 80kB but we need to fit data and code as well on there.
# For GAP9 we use 85kb because the L1 has a size of 126kB

x_index = np.where(np.abs(x_array - 50) < 0.1)[0][0]
x_index_v2 = np.where(np.abs(x_array_v2 - 50) < 0.1)[0][0]
print("GradientBoosting Mr.Wolf L1 capacity: ", x_array[x_index])
print("GradientBoosting Mr.Wolf L1 accuracy: ", y_array[x_index])
print("ExtraTrees Mr.Wolf L1 capacity: ", x_array_v2[x_index_v2])
print("ExtraTrees Mr.Wolf L1 accuracy: ", y_array_v2[x_index_v2])
increase_l1 = (y_array[x_index] - y_array_v2[x_index_v2])*100
print("Increase of accuracy for Mr.Wolf L1: ", increase_l1)

# Next L2:

x_index_l2 = np.where(np.abs(x_array - 512) < 5)[0][0]
x_index_l2_v2 = np.where(np.abs(x_array_v2 - 512) < 40)[0][1]
print("GradientBoosting Mr.Wolf L2 capacity: ", x_array[x_index_l2])
print("GradientBoosting Mr.Wolf L2 accuracy: ", y_array[x_index_l2])
print("ExtraTrees Mr.Wolf L2 capacity: ", x_array_v2[x_index_l2_v2])
print("ExtraTrees Mr.Wolf L2 accuracy: ", y_array_v2[x_index_l2_v2])
increase_l2 = (y_array[x_index_l2] - y_array_v2[x_index_l2_v2])*100
print("Increase of accuracy for Mr.Wolf L2: ", increase_l2)


# Next L1 for GAP9

x_index_gap9 = np.where(np.abs(x_array - 85) < 1)[0][0]
x_index_gap9_v2 = np.where(np.abs(x_array_v2 - 85) < 1)[0][0]
print("GradientBoosting GAP9 L1 capacity: ", x_array[x_index_gap9])
print("GradientBoosting GAP9 L1 accuracy: ", y_array[x_index_gap9])
print("ExtraTrees GAP9 L1 capacity: ", x_array_v2[x_index_gap9_v2])
print("ExtraTrees GAP9 L1 accuracy: ", y_array_v2[x_index_gap9_v2])
increase_gap9 = (y_array[x_index_gap9] - y_array_v2[x_index_gap9_v2])*100
print("Increase of accuracy for GAP9 L1: ", increase_gap9)

# Next L2 for GAP9

x_index_gap9_l2 = np.where(np.abs(x_array - 1500) < 50)[0][0]
x_index_gap9_l2_v2 = np.where(np.abs(x_array_v2 - 1500) < 100)[0][0]
print("GradientBoosting GAP9 L2 capacity: ", x_array[x_index_gap9_l2])
print("GradientBoosting GAP9 L2 accuracy: ", y_array[x_index_gap9_l2])
print("ExtraTrees GAP9 L2 capacity: ", x_array_v2[x_index_gap9_l2_v2])
print("ExtraTrees GAP9 L2 accuracy: ", y_array_v2[x_index_gap9_l2_v2])
increase_gap9_l2 = (y_array[x_index_gap9_l2] - y_array_v2[x_index_gap9_l2_v2])*100
print("Increase of accuracy for GAP9 L2: ", increase_gap9_l2)


fig, ax = plt.subplots(figsize=(12,8))
rects1 = ax.plot(x_array, y_array,color='blue',label='GradientBoostingClassifier')
rects1 = ax.plot(x_array_v2, y_array_v2,color='black',label='ExtraTreesClassifier')

l1 = 40
l2 = 512
plt.ylim([0.85, 0.95])
plt.xlim([0, 600])
ax.annotate('L2 Memory Capacity', xy=(l2, 0.89),  xycoords='data',
            xytext=(400, 0.89), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
ax.annotate('L1 Memory Capacity', xy=(l1, 0.88),  xycoords='data',
            xytext=(300, 0.88), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
plt.axvline(l2, c='red')
plt.axvline(l1, c='orange')
plt.xlabel('Size of model')
plt.ylabel('Accuracy [%]')
buffer = 0.001
#Print the increase in accuracy at L1 using GradientBoosting and ExtraTrees
ax.annotate("", xy=(50, y_array[x_index]-buffer), xytext=(50, y_array_v2[x_index_v2]+0.002),arrowprops=dict(color='green',arrowstyle="<->",linewidth=1.5))
ax.annotate("+"+str(round(increase_l1,2)) + "%", xy=(50, y_array[x_index]-buffer), xytext=(60, y_array_v2[x_index_v2]+0.01),color='green')
#Print the increase in accuracy at L2 using GradientBoosting and ExtraTrees
ax.annotate("", xy=(525, y_array[x_index_l2]), xytext=(525, y_array_v2[x_index_l2_v2]),arrowprops=dict(arrowstyle="<->",linewidth=1.5,color='green'))
ax.annotate("+"+str(round(increase_l2,2)) + "%", xy=(525, y_array[x_index_l2]), xytext=(527, y_array_v2[x_index_l2_v2]+0.003),color='green')
ax.legend()
plt.title("Mr. Wolf - GradientBoostingClassifier vs ExtraTreesClassifier")
plt.savefig("figures/gradient_vs_extra_mrwolf_binary.png")


fig, ax = plt.subplots(figsize=(8,6))
rects1 = ax.plot(x_array, y_array,color='blue',label='GradientBoostingClassifier')
rects1 = ax.plot(x_array_v2, y_array_v2,color='black',label='ExtraTreesClassifier')
l1 = 80
l2 = 1500
plt.ylim([0.85, 0.95])
plt.xlim([0, 1800])
ax.annotate('L2 Memory Capacity', xy=(l2, 0.90),  xycoords='data',
            xytext=(1000, 0.90), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
ax.annotate('L1 Memory Capacity', xy=(l1, 0.88),  xycoords='data',
            xytext=(600, 0.88), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
plt.axvline(l2, c='red')
plt.axvline(l1, c='green')
buffer = 0.001
#Print the increase in accuracy at L1 using GradientBoosting and ExtraTrees
ax.annotate("", xy=(100, y_array[x_index_gap9]-buffer), xytext=(100, y_array_v2[x_index_gap9_v2]+0.002),arrowprops=dict(color='green',arrowstyle="<->",linewidth=1.5))
ax.annotate("+"+str(round(increase_gap9,2)) + "%", xy=(100, y_array[x_index_gap9]-buffer), xytext=(110, y_array_v2[x_index_gap9_v2]+0.01),color='green')
#Print the increase in accuracy at L2 using GradientBoosting and ExtraTrees
ax.annotate("", xy=(1525, y_array[x_index_gap9_l2]), xytext=(1525, y_array_v2[x_index_gap9_l2_v2]),arrowprops=dict(arrowstyle="<->",linewidth=1.5,color='green'))
ax.annotate("+"+str(round(increase_gap9_l2,2)) + "%", xy=(1525, y_array[x_index_gap9_l2]), xytext=(1535, y_array_v2[x_index_gap9_l2_v2]+0.0015),color='green')
ax.legend()
plt.title("GAP9 - GradientBoostingClassifier vs ExtraTreesClassifier")
plt.savefig("figures/gradient_vs_extra_gap9_binary.png")


df = pd.read_pickle("results/gradient_results_mmc_32.pkl")
x_array_mc = np.load('results/kb_array_MC.npy')
y_array_mc = np.load('results/accuracy_array_MC.npy')
x_array_mc_32 = df[1].to_numpy()
y_array_mc_32 = df[0].to_numpy() 


#Find the increase in accuracy at L1 capacity and and L2 capacity for Mr.Wolf
# We say L1 capacity is 40 for mr wolf beacuse the L1 has a size of 80kB but we need to fit data and code as well on there.
# For GAP9 we use 85kb because the L1 has a size of 126kB

x_index = np.where(np.abs(x_array_mc_32 - 50) < 10)[0][0]
x_index_v2 = np.where(np.abs(x_array_mc - 50) < 10)[0][0]
print("GradientBoosting Mr.Wolf L1 capacity: ", x_array_mc_32[x_index])
print("GradientBoosting Mr.Wolf L1 accuracy: ", y_array_mc_32[x_index])
print("ExtraTrees Mr.Wolf L1 capacity: ", x_array_mc[x_index_v2])
print("ExtraTrees Mr.Wolf L1 accuracy: ", y_array_mc[x_index_v2])
increase_l1 = (y_array_mc_32[x_index] - y_array_mc[x_index_v2])*100
print("Increase of accuracy for Mr.Wolf L1: ", increase_l1)

# Next L2:

x_index_l2 = np.where(np.abs(x_array_mc_32 - 512) < 70)[0][0]
x_index_l2_v2 = np.where(np.abs(x_array_mc - 512) < 600)[0][1]
print("GradientBoosting Mr.Wolf L2 capacity: ", x_array_mc_32[x_index_l2])
print("GradientBoosting Mr.Wolf L2 accuracy: ", y_array_mc_32[x_index_l2])
print("ExtraTrees Mr.Wolf L2 capacity: ", x_array_mc[x_index_l2_v2])
print("ExtraTrees Mr.Wolf L2 accuracy: ", y_array_mc[x_index_l2_v2])
increase_l2 = (y_array_mc_32[x_index_l2] - y_array_mc[x_index_l2_v2])*100
print("Increase of accuracy for Mr.Wolf L2: ", increase_l2)


# Next L1 for GAP9

x_index_gap9 = np.where(np.abs(x_array_mc_32 - 85) < 10)[0][0]
x_index_gap9_v2 = np.where(np.abs(x_array_mc - 85) < 10)[0][0]
print("GradientBoosting GAP9 L1 capacity: ", x_array_mc_32[x_index_gap9])
print("GradientBoosting GAP9 L1 accuracy: ", y_array_mc_32[x_index_gap9])
print("ExtraTrees GAP9 L1 capacity: ", x_array_mc[x_index_gap9_v2])
print("ExtraTrees GAP9 L1 accuracy: ", y_array_mc[x_index_gap9_v2])
increase_gap9 = (y_array_mc_32[x_index_gap9] - y_array_mc[x_index_gap9_v2])*100
print("Increase of accuracy for GAP9 L1: ", increase_gap9)

# Next L2 for GAP9

x_index_gap9_l2 = np.where(np.abs(x_array_mc_32 - 1500) < 400)[0][0]
x_index_gap9_l2_v2 = np.where(np.abs(x_array_mc - 1500) < 400)[0][0]
print("GradientBoosting GAP9 L2 capacity: ", x_array_mc_32[x_index_gap9_l2])
print("GradientBoosting GAP9 L2 accuracy: ", y_array_mc_32[x_index_gap9_l2])
print("ExtraTrees GAP9 L2 capacity: ", x_array_mc[x_index_gap9_l2_v2])
print("ExtraTrees GAP9 L2 accuracy: ", y_array_mc[x_index_gap9_l2_v2])
increase_gap9_l2 = (y_array_mc_32[x_index_gap9_l2] - y_array_mc[x_index_gap9_l2_v2])*100
print("Increase of accuracy for GAP9 L2: ", increase_gap9_l2)


fig, ax = plt.subplots(figsize=(8,6))
rects1 = ax.plot(x_array_mc_32, y_array_mc_32,color='blue',label='GradientBoostingClassifier')
rects1 = ax.plot(x_array_mc, y_array_mc,color='black',label='ExtraTreesClassifier')
l1 = 40
l2 = 512
plt.ylim([0.85, 0.95])
plt.xlim([0, 600])
ax.annotate('L2 Memory Capacity', xy=(l2, 0.89),  xycoords='data',
            xytext=(400, 0.89), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
ax.annotate('L1 Memory Capacity', xy=(l1, 0.88),  xycoords='data',
            xytext=(300, 0.88), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
plt.axvline(l2, c='red')
plt.axvline(l1, c='orange')
buffer = 0.001
#Print the increase in accuracy at L1 using GradientBoosting and ExtraTrees
ax.annotate("", xy=(50, y_array_mc_32[x_index]), xytext=(50, y_array_mc[x_index_v2]-0.001),arrowprops=dict(color='green',arrowstyle="<->",linewidth=1.5))
ax.annotate("+"+str(round(increase_l1,2)) + "%", xy=(50, y_array_mc_32[x_index]), xytext=(60, y_array_mc[x_index_v2]+0.002),color='green')
#Print the increase in accuracy at L2 using GradientBoosting and ExtraTrees
ax.annotate("", xy=(525, y_array_mc_32[x_index_l2]+0.001), xytext=(525, y_array_mc[x_index_l2_v2]),arrowprops=dict(arrowstyle="<->",linewidth=1.5,color='green'))
ax.annotate("+"+str(round(increase_l2,2)) + "%", xy=(525, y_array_mc_32[x_index_l2]), xytext=(527, y_array_mc[x_index_l2_v2]+0.002),color='green')
ax.legend()
plt.title("Mr. Wolf - GradientBoostingClassifier vs ExtraTreesClassifier")
plt.savefig("figures/gradient_vs_extra_mrwolf_mc.png")


fig, ax = plt.subplots(figsize=(8,6))
rects1 = ax.plot(x_array_mc_32, y_array_mc_32,color='blue',label='GradientBoostingClassifier')
rects1 = ax.plot(x_array_mc, y_array_mc,color='black',label='ExtraTreesClassifier')
l1 = 80
l2 = 1500
plt.ylim([0.85, 0.95])
plt.xlim([0, 1800])
ax.annotate('L2 Memory Capacity', xy=(l2, 0.92),  xycoords='data',
            xytext=(1000, 0.92), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
ax.annotate('L1 Memory Capacity', xy=(l1, 0.88),  xycoords='data',
            xytext=(600, 0.88), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
plt.axvline(l2, c='red')
plt.axvline(l1, c='green')
buffer = 0.001
#Print the increase in accuracy at L1 using GradientBoosting and ExtraTrees
ax.annotate("", xy=(100, y_array_mc_32[x_index_gap9]+0.001), xytext=(100, y_array_mc[x_index_gap9_v2]),arrowprops=dict(color='green',arrowstyle="<->",linewidth=1.5))
ax.annotate("+"+str(round(increase_gap9,2)) + "%", xy=(100, y_array_mc_32[x_index_gap9]-buffer), xytext=(110, y_array_mc[x_index_gap9_v2]+0.0035),color='green')
#Print the increase in accuracy at L2 using GradientBoosting and ExtraTrees
ax.annotate("", xy=(1525, y_array_mc_32[x_index_gap9_l2]), xytext=(1525, y_array_mc[x_index_gap9_l2_v2]),arrowprops=dict(arrowstyle="<->",linewidth=1.5,color='green'))
ax.annotate("+"+str(round(increase_gap9_l2,2)) + "%", xy=(1525, y_array_mc_32[x_index_gap9_l2]), xytext=(1535, y_array_mc[x_index_gap9_l2_v2]+0.005),color='green')
ax.legend()
plt.title("GAP9 - GradientBoostingClassifier vs ExtraTreesClassifier")
plt.savefig("figures/gradient_vs_extra_gap9_mc.png")


df = pd.read_pickle("results/gradient_results_mmo_32.pkl")
x_array_mc = np.load('results/kb_array_MMC.npy')
y_array_mc = np.load('results/accuracy_array_MMC.npy')
x_array_mc_32 = df[1].to_numpy()
y_array_mc_32 = df[0].to_numpy() 


#Find the increase in accuracy at L1 capacity and and L2 capacity for Mr.Wolf
# We say L1 capacity is 40 for mr wolf beacuse the L1 has a size of 80kB but we need to fit data and code as well on there.
# For GAP9 we use 85kb because the L1 has a size of 126kB

x_index = np.where(np.abs(x_array_mc_32 - 50) < 10)[0][0]
x_index_v2 = np.where(np.abs(x_array_mc - 50) < 10)[0][0]
print("GradientBoosting Mr.Wolf L1 capacity: ", x_array_mc_32[x_index])
print("GradientBoosting Mr.Wolf L1 accuracy: ", y_array_mc_32[x_index])
print("ExtraTrees Mr.Wolf L1 capacity: ", x_array_mc[x_index_v2])
print("ExtraTrees Mr.Wolf L1 accuracy: ", y_array_mc[x_index_v2])
increase_l1 = (y_array_mc_32[x_index] - y_array_mc[x_index_v2])*100
print("Increase of accuracy for Mr.Wolf L1: ", increase_l1)

# Next L2:

x_index_l2 = np.where(np.abs(x_array_mc_32 - 512) < 70)[0][0]
x_index_l2_v2 = np.where(np.abs(x_array_mc - 512) < 600)[0][1]
print("GradientBoosting Mr.Wolf L2 capacity: ", x_array_mc_32[x_index_l2])
print("GradientBoosting Mr.Wolf L2 accuracy: ", y_array_mc_32[x_index_l2])
print("ExtraTrees Mr.Wolf L2 capacity: ", x_array_mc[x_index_l2_v2])
print("ExtraTrees Mr.Wolf L2 accuracy: ", y_array_mc[x_index_l2_v2])
increase_l2 = (y_array_mc_32[x_index_l2] - y_array_mc[x_index_l2_v2])*100
print("Increase of accuracy for Mr.Wolf L2: ", increase_l2)


# Next L1 for GAP9

x_index_gap9 = np.where(np.abs(x_array_mc_32 - 85) < 10)[0][0]
x_index_gap9_v2 = np.where(np.abs(x_array_mc - 85) < 10)[0][0]
print("GradientBoosting GAP9 L1 capacity: ", x_array_mc_32[x_index_gap9])
print("GradientBoosting GAP9 L1 accuracy: ", y_array_mc_32[x_index_gap9])
print("ExtraTrees GAP9 L1 capacity: ", x_array_mc[x_index_gap9_v2])
print("ExtraTrees GAP9 L1 accuracy: ", y_array_mc[x_index_gap9_v2])
increase_gap9 = (y_array_mc_32[x_index_gap9] - y_array_mc[x_index_gap9_v2])*100
print("Increase of accuracy for GAP9 L1: ", increase_gap9)

# Next L2 for GAP9

x_index_gap9_l2 = np.where(np.abs(x_array_mc_32 - 1500) < 400)[0][0]
x_index_gap9_l2_v2 = np.where(np.abs(x_array_mc - 1500) < 400)[0][0]
print("GradientBoosting GAP9 L2 capacity: ", x_array_mc_32[x_index_gap9_l2])
print("GradientBoosting GAP9 L2 accuracy: ", y_array_mc_32[x_index_gap9_l2])
print("ExtraTrees GAP9 L2 capacity: ", x_array_mc[x_index_gap9_l2_v2])
print("ExtraTrees GAP9 L2 accuracy: ", y_array_mc[x_index_gap9_l2_v2])
increase_gap9_l2 = (y_array_mc_32[x_index_gap9_l2] - y_array_mc[x_index_gap9_l2_v2])*100
print("Increase of accuracy for GAP9 L2: ", increase_gap9_l2)


fig, ax = plt.subplots(figsize=(8,6))
rects1 = ax.plot(x_array_mc_32, y_array_mc_32,color='blue',label='GradientBoostingClassifier')
rects1 = ax.plot(x_array_mc, y_array_mc,color='black',label='ExtraTreesClassifier')
l1 = 40
l2 = 512
plt.ylim([0.85, 0.95])
plt.xlim([0, 600])
ax.annotate('L2 Memory Capacity', xy=(l2, 0.89),  xycoords='data',
            xytext=(400, 0.89), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
ax.annotate('L1 Memory Capacity', xy=(l1, 0.88),  xycoords='data',
            xytext=(300, 0.88), textcoords='data',
            arrowprops=dict(facecolor='black', shrink=0.05),
            horizontalalignment='right', verticalalignment='top',
            )
plt.axvline(l2, c='red')
plt.axvline(l1, c='orange')
buffer = 0.001
#Print the increase in accuracy at L1 using GradientBoosting and ExtraTrees
ax.annotate("", xy=(50, y_array_mc_32[x_index]), xytext=(50, y_array_mc[x_index_v2]-0.001),arrowprops=dict(color='green',arrowstyle="<->",linewidth=1.5))
ax.annotate("+"+str(round(increase_l1,2)) + "%", xy=(50, y_array_mc_32[x_index]), xytext=(60, y_array_mc[x_index_v2]+0.002),color='green')
#Print the increase in accuracy at L2 using GradientBoosting and ExtraTrees
ax.annotate("", xy=(525, y_array_mc_32[x_index_l2]+0.001), xytext=(525, y_array_mc[x_index_l2_v2]),arrowprops=dict(arrowstyle="<->",linewidth=1.5,color='green'))
ax.annotate("+"+str(round(increase_l2,2)) + "%", xy=(525, y_array_mc_32[x_index_l2]), xytext=(527, y_array_mc[x_index_l2_v2]+0.002),color='green')
ax.legend()
plt.title("Mr. Wolf - GradientBoostingClassifier vs ExtraTreesClassifier")
plt.savefig("figures/gradient_vs_extra_mrwolf_mmo.png")


