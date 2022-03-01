from datetime import time
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from scipy.optimize import curve_fit

def exp_func(x,a,b,c):
    return b*np.exp(a*x)+c

rootdir = "/mnt/e/noisetimedata/noisedata_hist/"
rootdir_time = "/mnt/e/noisetimedata/timeseq/"
targetdir = "/mnt/e/noisetimedata/noisedata_hist_new/"
figdir = "/home/zerui_liu/work/KMTNet_breaker/testnoisemodelfig/"

mylog = open('recode.log', mode='w', encoding='utf-8')

fileslist = os.listdir(rootdir)
fileslist_time = os.listdir(rootdir_time)


def killbadtimedata():
    for i in range(len(fileslist_time)):
        timedata = np.load(rootdir_time+fileslist_time[i],allow_pickle=True)
        print(fileslist_time[i])
        print(len(timedata))
        timedata_new = []
        
        for j in range(len(timedata)):
            if len(timedata[j])<10:
                print("length of timeseq too short",timedata[j],"posi: ",j,file=mylog)
        
        '''
        

        for j in range(len(timedata)):
            if 
        '''
        
            

def killbaddata():
    llim = []
    rlim = []
    for i in range(len(fileslist)):
        alert = 0

        mag_err_data = np.load(rootdir+fileslist[i],allow_pickle=True)
        print(fileslist[i])

        # print(fileslist[i],", (leftlim, rightlim):",mag_err_data[0][0],mag_err_data[0][1],file=mylog)
        range_message = mag_err_data[0] 
        hist_message = mag_err_data[1]
        hist_bins = mag_err_data[2]
        llim.append(range_message[0])
        rlim.append(range_message[1])
        for j in range(len(hist_message)):
            histbin = np.array(hist_bins[j])
            
            if (np.isnan(histbin)).any() or (np.isinf(histbin)).any():
                print(fileslist[i],"nan/inf alert: ",j,file=mylog)
            
            if (histbin<0).any():
                print(fileslist[i],"negative alert: ",j,file=mylog)

            if (np.isnan(1/histbin)).any():
                print(fileslist[i],"zero nan alert: ",j,file=mylog)
            
            if (np.isinf(1/histbin)).any():
                print(fileslist[i],"zero inf alert: ",j,file=mylog)
                
                alert += 1
                histbin_dealed = list(histbin[histbin>0])
                hist_bins[j] = histbin_dealed
                hist_message[j] = len(histbin_dealed)
        if alert > 0:
            data_array = np.array([range_message,hist_message,hist_bins],dtype=object)
            np.save(targetdir+fileslist[i], data_array, allow_pickle = True)
    print(np.max(llim))
    print(np.min(rlim))

def fit(index):
    mag_err_data = np.load(rootdir+fileslist[index],allow_pickle=True)

    range_message = mag_err_data[0] 
    hist_message = mag_err_data[1]
    hist_bins = mag_err_data[2]

    mags = []
    errs_mean = []
    errs_std = []

    for i in range(len(hist_message)):
        if hist_message[i] > 0:
            errs_mean.append(np.mean(hist_bins[i]))
            errs_std.append(np.std(hist_bins[i]))
            mags.append(range_message[0]+(range_message[1]-range_message[0])*i/len(hist_message))

    popt, pcov = curve_fit(exp_func,mags,errs_mean)#,bounds=(left_bound,right_bound))
    lc_fit = exp_func(np.array(mags),*popt)

    plt.figure()
    plt.subplot(211)
    plt.scatter(mags,errs_mean,s=3)
    plt.plot(mags,lc_fit)
    plt.xlabel("mag")
    plt.ylabel("err_mean")
    plt.subplot(212)
    plt.scatter(mags,errs_std,s=3)
    plt.xlabel("mag")
    plt.ylabel("err_std")
    plt.savefig(figdir+fileslist[index][:-4]+".png")
    plt.close()

    return popt

def fix_zero(index,fixnum = 10):
    mag_err_data = np.load(rootdir+fileslist[index],allow_pickle=True)

    range_message = mag_err_data[0] 
    hist_message = np.array(mag_err_data[1]).astype(np.int)
    hist_bins = mag_err_data[2]

    mags = []
    errs_mean = []
    errs_std = []

    for i in range(len(hist_message)):
        if hist_message[i] > 0:
            errs_mean.append(np.mean(hist_bins[i]))
            errs_std.append(np.std(hist_bins[i]))
            mags.append(range_message[0]+(range_message[1]-range_message[0])*i/len(hist_message))

    popt, pcov = curve_fit(exp_func,mags,errs_mean)#,bounds=(left_bound,right_bound))

    for i in range(len(hist_message)):
        if hist_message[i] <= 0:
            hist_message[i] = fixnum
            mags_posi = range_message[0]+(range_message[1]-range_message[0])*i/len(hist_message)
            hist_bins[i] = list((0.1+0.9*np.random.uniform(size=fixnum))*exp_func(mags_posi,*popt))
    data_array = np.array([range_message,list(hist_message),hist_bins],dtype=object)
    np.save(targetdir+fileslist[index], data_array, allow_pickle = True)


killbadtimedata()

mylog.close()