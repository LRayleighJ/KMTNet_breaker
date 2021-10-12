import MulensModel as mm
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import re
import pandas as pd
import KMTNet_breaker.loaddata.loadKMTdata as loaddata
from scipy.optimize import curve_fit


def mag_cal(t,tE,t0,u0,fs,fb,m0):
    u = np.sqrt(((t-t0)/tE)**2+u0**2)
    A = (u**2+2)/(u*np.sqrt(u**2+4))
    return m0-2.5*np.log10(fs*A+fb)

def chi_square(single,binary,sigma):
    return np.sum(np.power((single-binary)/sigma,2))

'''
args,data,_,_,_= loaddata.getKMTdata(year=2018,posi=1046,cut=1,cutratio=2)
print(args)

'''
for index in range(59000,59000+100):
    datadir = list(np.load("./sample_simu_2/%d.npy"%(index,), allow_pickle=True))
        
    labels = np.array(datadir[0],dtype=np.float64)
    print(labels)

    mag = np.array(datadir[3],dtype=np.float64)
    time = np.array(datadir[1],dtype=np.float64)
    errorbar = np.array(datadir[4],dtype=np.float64)

    # print(data_input.shape)
    # [u_0, rho, q, s, alpha, t_E]

    lg_q = np.log10(labels[2])
    lg_s = np.log10(labels[3])
    alpha = labels[4]
    u0 = labels[0]


    t_0_KMT = 0
    t_E_KMT = labels[-2]
    u_0_KMT = labels[0]
    Ibase_KMT = np.mean(np.sort(mag)[-50:])

    left_bound = [0.5*t_E_KMT,t_0_KMT-0.5*t_E_KMT,u_0_KMT-1,1-0.5,0-0.5,0.7*Ibase_KMT]
    right_bound = [1.5*t_E_KMT,t_0_KMT+0.5*t_E_KMT,u_0_KMT+1,1+0.5,0+0.5,1.3*Ibase_KMT]

    popt, pcov = curve_fit(mag_cal,time,mag,bounds=(left_bound,right_bound))
    lc_fit = mag_cal(time,*popt)

    chi_s = chi_square(lc_fit,mag,errorbar)
    print(popt)
    print(chi_s)

    plt.figure(figsize=[20,18])
    plt.subplot(411)
    plt.errorbar(time,mag,yerr=errorbar,fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
    plt.plot(time,lc_fit)
    plt.xlabel("t/HJD-2450000",fontsize=16)
    plt.ylabel("Magnitude",fontsize=16)
    plt.gca().invert_yaxis()
    plt.title("$\Delta \chi^2$: "+str(chi_s)+" label: "+str(int(labels[-1])))
    plt.subplot(412)
    plt.errorbar(time,mag-lc_fit,yerr=errorbar,fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
    
    plt.subplot(413)
    plt.scatter(time,((mag-lc_fit)/errorbar)**2)
    

    plt.subplot(414)
    plt.hist(((mag-lc_fit)/errorbar)**2,bins=100)


    plt.savefig("./fig_simu_2/testfit_simu_%d.png"%(index,))
    plt.close()