import numpy as np
import matplotlib.pyplot as plt
import os
import random

rootdir = "/mnt/e/noisetimedata/noisedata_hist/"
rootdir_time = "/mnt/e/noisetimedata/timeseq/"

# mag_err_data = np.load(rootdir+os.listdir(rootdir)[random.randint(0,24)],allow_pickle=True)
time_data = np.load(rootdir_time+os.listdir(rootdir_time)[random.randint(0,24)],allow_pickle=True)

# list0 = mag_err_data[0]
# list1 = mag_err_data[1]
# 
# list2 = mag_err_data[2]
# list3 = mag_err_data[3]

time_data_index = [x for x in range(len(time_data)) if len(time_data[x]) > 1000]

print(time_data_index)