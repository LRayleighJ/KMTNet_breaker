import os
import numpy as np
import pandas as pd
import KMTNet_breaker.loaddata.loadKMTdata as loaddata
import matplotlib.pyplot as plt

def cut_data(x,threshold=3000):
    if x < threshold:
        return x
    else:
        return threshold

targetdir = "/mnt/e/KMT_catalog/"

years_list = [2017, 2018,2019,2020]
num_events = [100,100,100,100]#[2817, 2781, 3303, 894]

static_length_time = []
static_t_E = []
static_m_0 = []
static_density = []

for kk in range(len(years_list)):
    for i in np.arange(num_events[kk]):
        i_blyat = i+1

        years = years_list[kk]

        file_number = "%04d" % i_blyat
        folder_name = str(years)+ '_' + file_number
        try:
            args,data,_,_,_= loaddata.getKMTdata(year=years,posi=i_blyat,cut=1,cutratio=2,FWHM_threshold=7,sky_threshold=10000)
            print(args)

            time = data[0]
            mag = data[1]
            errorbar = data[2]

            t_0_KMT = args[-4]
            t_E_KMT = args[-3]
            u_0_KMT = args[-2]
            Ibase_KMT = args[-1]

            if int(len(time)*(4*t_E_KMT)/(time[-1]-time[0]))<1000:
                static_length_time.append(cut_data(int(len(time)*(4*t_E_KMT)/(time[-1]-time[0]))))
            static_t_E.append(t_E_KMT)
            static_m_0.append(Ibase_KMT)
            static_density.append(len(time)/(time[-1]-time[0]))
        except:
            continue
print("final",len(static_length_time))


plt.figure(figsize=(12,30))
plt.subplot(511)
plt.hist(static_length_time,bins=100)
plt.xlabel("number of datapoint")
plt.title("number of datapoint")
plt.subplot(512)
plt.hist(static_t_E,bins=1000)
plt.xlabel("t_E")
plt.title("t_E")
plt.subplot(513)
plt.hist(static_m_0,bins=1000)
plt.xlabel("m_0")
plt.title("m_0")
plt.subplot(514)
plt.hist(static_density,bins=1000)
plt.xlabel("density")
plt.title("density")
plt.subplot(515)
plt.hist(static_density,bins=1000)
plt.xlabel("density")
plt.title("density")


plt.savefig("static.png")


