import MulensModel as mm
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import re

def magnitude_tran(magni,m_0=18):
    return m_0 - 2.5*np.log10(magni)

def trajectory(timedomain,q,s,u0,alpha,te,rho):
    bl_model = mm.Model({'t_0': 0, 'u_0': u0,'t_E': te, 'rho': rho, 'q': q, 's': s,'alpha': alpha})
    bl_model.set_default_magnification_method("VBBL")
    
    mag = bl_model.magnification(timedomain)
    
    caustic = mm.Caustics(s=s, q=q)
    X,Y = caustic.get_caustics(n_points=2000)

    trace_x = -np.sin(alpha*np.pi/180)*u0+timedomain/te*np.cos(alpha*np.pi/180)
    trace_y = np.cos(alpha*np.pi/180)*u0+timedomain/te*np.sin(alpha*np.pi/180)

    plt.figure(figsize=(16,5))
    plt.subplot(121)
    plt.scatter(X,Y,s=1,c="b")
    plt.plot(trace_x,trace_y,c="g")
    plt.xlabel(r"$\theta_x$")
    plt.ylabel(r"$\theta_y$")
    plt.axis("scaled")
    plt.grid()
    

    plt.subplot(122)
    plt.scatter(timedomain,magnitude_tran(mag,0),s=20, marker="x")
    plt.xlabel("t(HJD)")
    plt.ylabel("mag")
    plt.gca().invert_yaxis()

    plt.grid()
    plt.show()
    plt.close() 

    return timedomain,magnitude_tran(mag,0)

def cut(t_0,t_e,data):
    new_data = []
    for array_sample in data:
        if array_sample[0]<t_0+2*t_e and array_sample[0]>t_0-2*t_e:
            if array_sample[-3]<8:
                new_data.append(array_sample)
    return np.array(new_data)

# getfile

def getKMTIfilelist(rootdir = "/mnt/e/KMT_catalog/",year=2018,posi=1):
    path = rootdir+"%d_%04d/"%(year,posi,)
    filelist_origin = os.listdir(path)

    filelist = [filename for filename in filelist_origin if re.search("KMT.*_I.pysis",filename)]
    filelist_A = [filename for filename in filelist if filename[3]=="A"]
    filelist_C = [filename for filename in filelist if filename[3]=="C"]
    filelist_S = [filename for filename in filelist if filename[3]=="S"]

    return filelist_A,filelist_C,filelist_S



def DrawKMTdata(rootdir = "/mnt/e/KMT_catalog/",year=2018,posi=1):
    path = rootdir+"%d_%04d/"%(year,posi,)
    filelistI_A,filelistI_C,filelistI_S = getKMTIfilelist(rootdir,year,posi)
    datas_A = [np.loadtxt(path+filename).T for filename in filelistI_A]
    datas_C = [np.loadtxt(path+filename).T for filename in filelistI_C]
    datas_S = [np.loadtxt(path+filename).T for filename in filelistI_S]
    data_A = np.hstack(datas_A)
    data_C = np.hstack(datas_C)
    data_S = np.hstack(datas_S)


    print(data_A.shape)

    data = np.c_[data_A,data_C,data_S]
    time = data[0]
    mag = data[3]
    errorbar = data[4]

    order = np.argsort(time)
    time = time[order]
    mag = mag[order]
    errorbar = errorbar[order]

    print(len(data_A[0])+len(data_C[0])+len(data_S[0]))

    plt.figure(figsize=(21,15))

    ax1 = plt.subplot2grid((7,1), (0,0), rowspan=4)
    plt.scatter(data_A[0],data_A[3],s=6,alpha=0.5,label="KMTA")
    plt.errorbar(data_A[0],data_A[3],yerr=data_A[4],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
    plt.scatter(data_C[0],data_C[3],s=6,alpha=0.5,label="KMTC")
    plt.errorbar(data_C[0],data_C[3],yerr=data_C[4],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
    plt.scatter(data_S[0],data_S[3],s=6,alpha=0.5,label="KMTS")
    plt.errorbar(data_S[0],data_S[3],yerr=data_S[4],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)

    # plt.plot(time_simulate+t0,mag_simulate+19.59,c="black",linewidth=1,label="Best Fit")
    # plt.plot(time_simulate_2+t0,mag_simulate_2+19.59,c="r",linewidth=1,label="MDN Predicted")

    plt.xlabel("t/HJD-2450000",fontsize=16)
    plt.ylabel("Magnitude",fontsize=16)
    plt.legend()
    
    plt.gca().invert_yaxis()

    ax2 = plt.subplot2grid((7,1), (4, 0))
    plt.scatter(data_A[0],data_A[-3],s=6,alpha=0.5,label="KMTA")
    plt.scatter(data_C[0],data_C[-3],s=6,alpha=0.5,label="KMTC")
    plt.scatter(data_S[0],data_S[-3],s=6,alpha=0.5,label="KMTS")
    plt.legend()
    plt.xlabel("t/HJD-2450000")
    plt.ylabel("FWHM")
    ax3 = plt.subplot2grid((7,1), (5, 0))
    plt.scatter(data_A[0],data_A[-2],s=6,alpha=0.5,label="KMTA")
    plt.scatter(data_C[0],data_C[-2],s=6,alpha=0.5,label="KMTC")
    plt.scatter(data_S[0],data_S[-2],s=6,alpha=0.5,label="KMTS")
    plt.legend()
    plt.xlabel("t/HJD-2450000")
    plt.ylabel("Sky")
    ax4 = plt.subplot2grid((7,1), (6, 0))
    plt.scatter(data_A[0],data_A[-1],s=6,alpha=0.5,label="KMTA")
    plt.scatter(data_C[0],data_C[-1],s=6,alpha=0.5,label="KMTC")
    plt.scatter(data_S[0],data_S[-1],s=6,alpha=0.5,label="KMTS")
    plt.legend()
    plt.xlabel("t/HJD-2450000")
    plt.ylabel("secz")
    '''
    '''
    plt.suptitle("KMT-%d-BLG-%04d"%(year,posi,),fontsize=32)

    plt.show()

    plt.savefig("test_realKMT.png")
    plt.close()

DrawKMTdata(posi=2)