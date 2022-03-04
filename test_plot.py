import os
import numpy as np
import pandas as pd
import KMTNet_breaker.loaddata.loadKMTdata as loaddata
import KMTNet_breaker.fit.fitKMTdata as fitdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import MulensModel as mm

year=2019
posi=1

'''
data_args = pd.DataFrame(data=np.load("KMT_args.npy",allow_pickle=True))
print(data_args)
loaddata.DrawKMTdata(year=2019,posi=1715,cut=1,cutratio=0.5,sky_threshold=1500,FWHM_threshold=5)
'''
args, data, _,_,_= loaddata.getKMTdata(year=year,posi=posi,cut=1,cutratio=2,sky_threshold=1000,FWHM_threshold=10)

time = data[0]
mag = data[1]
errorbar = data[2]

print(args[-4:])

args_fit,chi_s = fitdata.fit_single(data[0],data[1],data[2],args[-4:])
print(args_fit)

plt.figure(figsize = (16,8))
plt.scatter(time,mag,s=6,alpha=0.5)
plt.errorbar(time,mag,yerr=errorbar,fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
plt.plot(time,fitdata.mag_cal(time,*args_fit))
plt.xlabel("t/HJD-2450000",fontsize=16)
plt.ylabel("Magnitude",fontsize=16)
# plt.legend()
plt.gca().invert_yaxis()

plt.suptitle("KMT-%d-BLG-%04d, $\chi^2$=%.3f"%(year,posi,chi_s))

plt.savefig("test_minimize_fit.png")

