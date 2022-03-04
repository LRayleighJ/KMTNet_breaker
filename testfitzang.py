import os
import numpy as np
import pandas as pd
import KMTNet_breaker.loaddata.loadKMTdata as loaddata
import KMTNet_breaker.fit.fitKMTdata as fitdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MulensModel as mm
from scipy import optimize

year=2019
posi=3

filepathzang = "/mnt/e/KMT_res/2019/KB19%04d.res"%(posi,)

def f_1(x, A, B):
    return A * x + B

#resdata = np.loadtxt()

args, data_flux, _,_,_= loaddata.getKMTfluxdata(year=year,posi=posi,cut=1,cutratio=2,sky_threshold=1000,FWHM_threshold=10)
args_mag, data_mag, _,_,_= loaddata.getKMTdata(year=year,posi=posi,cut=1,cutratio=2,sky_threshold=1000,FWHM_threshold=10)

time = data_flux[0]
flux = data_flux[1]
mag = data_mag[1]

m0 = args[-1]

print(args[-4:])
A1, B1 = optimize.curve_fit(f_1,flux, 10**(-2/5*mag))[0]
print(A1,B1)
print(3631*A1)
print(B1/10**(-2/5*m0))
print(-5/2*np.log10(B1))
print(m0)

plt.figure(figsize = (16,8))
plt.scatter(flux,10**(-2/5*mag),s=6,alpha=0.5)
plt.plot(flux,f_1(flux,A1,B1))
plt.savefig("test_minimize_fit.png")

