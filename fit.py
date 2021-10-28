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


args,data,_,_,_= loaddata.getKMTdata(year=2018,posi=1046,cut=1,cutratio=2,FWHM_threshold=4,sky_threshold=500)
print(args)

time = data[0]
mag = data[1]
errorbar = data[2]

t_0_KMT = args[-4]
t_E_KMT = args[-3]
u_0_KMT = args[-2]
Ibase_KMT = args[-1]

left_bound = [0.5*t_E_KMT,t_0_KMT-0.5*t_E_KMT,u_0_KMT-1,1-0.5,0-0.5,0.7*Ibase_KMT]
right_bound = [1.5*t_E_KMT,t_0_KMT+0.5*t_E_KMT,u_0_KMT+1,1+0.5,0+0.5,1.3*Ibase_KMT]

popt, pcov = curve_fit(mag_cal,time,mag,bounds=(left_bound,right_bound))
lc_fit = mag_cal(time,*popt)

chi_s = chi_square(lc_fit,mag,errorbar)
print(popt)
print(chi_s)

plt.figure(figsize=[20,14])
plt.errorbar(time,mag,yerr=errorbar,fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
plt.plot(time,lc_fit)
plt.xlabel("t/HJD-2450000",fontsize=16)
plt.ylabel("Magnitude",fontsize=16)
plt.gca().invert_yaxis()
plt.savefig("testfit.png")