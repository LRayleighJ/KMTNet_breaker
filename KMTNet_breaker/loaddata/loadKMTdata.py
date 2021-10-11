import MulensModel as mm
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import re
import pandas as pd

def magnitude_tran(magni,m_0=18):
    return m_0 - 2.5*np.log10(magni)

def single_amptitude(t,t_0,t_E,u_0):
    u = np.sqrt(((t-t_0)/t_E)**2+u_0**2)
    
    amp = (u**2+2)/(u*np.sqrt(u**2+4))
    return amp

def single_mag(t,t_0,t_E,u_0,m_0):
    return magnitude_tran(single_amptitude(t,t_0,t_E,u_0),m_0)

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

def getKMTdata(rootdir = "/mnt/e/KMT_catalog/",year=2018,posi=1,cut=0):
    data_args = pd.DataFrame(data=np.load("KMT_args.npy",allow_pickle=True))
    KMT_official_args = data_args.loc[data_args["index"]=="%d_%04d"%(year,posi,)].values[0]
    t_0_KMT = KMT_official_args[-4]
    t_E_KMT = KMT_official_args[-3]
    u_0_KMT = KMT_official_args[-2]
    Ibase_KMT = KMT_official_args[-1]

    path = rootdir+"%d_%04d/"%(year,posi,)
    filelistI_A,filelistI_C,filelistI_S = getKMTIfilelist(rootdir,year,posi)
    datas_A = [np.loadtxt(path+filename).T for filename in filelistI_A]
    datas_C = [np.loadtxt(path+filename).T for filename in filelistI_C]
    datas_S = [np.loadtxt(path+filename).T for filename in filelistI_S]
    data_A = np.hstack(datas_A)
    data_C = np.hstack(datas_C)
    data_S = np.hstack(datas_S)


    # print(data_A.shape)

    data = np.c_[data_A,data_C,data_S]
    time = np.linspace(np.min(data[0]),np.max(data[0]),1000)
    mag = data[3]
    errorbar = data[4]

    if cut != 0:
        time_select_index = np.argwhere((time<t_0_KMT+2*t_E_KMT)&(time>t_0_KMT-2*t_E_KMT)).T[0]
        time = time[time_select_index]
        mag = mag[time_select_index]
        errorbar = errorbar[time_select_index]
        print(len(time))
        data_A_index = np.argwhere((data_A[0]<t_0_KMT+2*t_E_KMT)&(data_A[0]>t_0_KMT-2*t_E_KMT)).T[0]
        data_C_index = np.argwhere((data_C[0]<t_0_KMT+2*t_E_KMT)&(data_C[0]>t_0_KMT-2*t_E_KMT)).T[0]
        data_S_index = np.argwhere((data_S[0]<t_0_KMT+2*t_E_KMT)&(data_S[0]>t_0_KMT-2*t_E_KMT)).T[0]

        order = np.argsort(time)
        time = time[order]
        mag = mag[order]
        errorbar = errorbar[order]

        dataA_sort = np.array([data_A[0][data_A_index],data_A[3][data_A_index],data_A[4][data_A_index]])
        dataC_sort = np.array([data_C[0][data_C_index],data_C[3][data_C_index],data_C[4][data_C_index]])
        dataS_sort = np.array([data_S[0][data_S_index],data_S[3][data_S_index],data_S[4][data_S_index]])
        return np.array([time,mag,errorbar]), dataA_sort, dataC_sort, dataS_sort 
    else:
        order = np.argsort(time)
        time = time[order]
        mag = mag[order]
        errorbar = errorbar[order]

        dataA_sort = np.array([data_A[0],data_A[3],data_A[4]])
        dataC_sort = np.array([data_C[0],data_C[3],data_C[4]])
        dataS_sort = np.array([data_S[0],data_S[3],data_S[4]])

        return np.array([time,mag,errorbar]), dataA_sort, dataC_sort, dataS_sort
    



def DrawKMTdata(rootdir = "/mnt/e/KMT_catalog/",year=2018,posi=1,cut=0,fit=0):
    data_args = pd.DataFrame(data=np.load("KMT_args.npy",allow_pickle=True))
    KMT_official_args = data_args.loc[data_args["index"]=="%d_%04d"%(year,posi,)].values[0]
    t_0_KMT = KMT_official_args[-4]
    t_E_KMT = KMT_official_args[-3]
    u_0_KMT = KMT_official_args[-2]
    Ibase_KMT = KMT_official_args[-1]
    print(KMT_official_args)
    print(t_0_KMT)
    print(t_E_KMT)

    
    path = rootdir+"%d_%04d/"%(year,posi,)
    filelistI_A,filelistI_C,filelistI_S = getKMTIfilelist(rootdir,year,posi)
    datas_A = [np.loadtxt(path+filename).T for filename in filelistI_A]
    datas_C = [np.loadtxt(path+filename).T for filename in filelistI_C]
    datas_S = [np.loadtxt(path+filename).T for filename in filelistI_S]
    data_A = np.hstack(datas_A)
    data_C = np.hstack(datas_C)
    data_S = np.hstack(datas_S)


    # print(data_A.shape)

    data = np.c_[data_A,data_C,data_S]
    time = np.linspace(np.min(data[0]),np.max(data[0]),1000)
    mag = data[3]
    errorbar = data[4]
    print(len(time))

    if fit:
        mag_fit = single_mag(t=time,t_0=t_0_KMT,t_E=t_E_KMT,m_0=21.31,u_0=u_0_KMT)
        #mulensmodel
        my_pspl_model = mm.Model({'t_0': t_0_KMT, 'u_0': u_0_KMT, 't_E': t_E_KMT})
        mag_fit_mm = magnitude_tran(my_pspl_model.magnification(time),m_0=21.31)
    # select

    if cut != 0:
        time_select_index = np.argwhere((time<t_0_KMT+2*t_E_KMT)&(time>t_0_KMT-2*t_E_KMT)).T[0]
        time = time[time_select_index]
        mag = mag[time_select_index]
        errorbar = errorbar[time_select_index]
        print(len(time))
        data_A_index = np.argwhere((data_A[0]<t_0_KMT+2*t_E_KMT)&(data_A[0]>t_0_KMT-2*t_E_KMT)).T[0]
        data_C_index = np.argwhere((data_C[0]<t_0_KMT+2*t_E_KMT)&(data_C[0]>t_0_KMT-2*t_E_KMT)).T[0]
        data_S_index = np.argwhere((data_S[0]<t_0_KMT+2*t_E_KMT)&(data_S[0]>t_0_KMT-2*t_E_KMT)).T[0]

        order = np.argsort(time)
        time = time[order]
        mag = mag[order]
        errorbar = errorbar[order]

        
        plt.figure(figsize=(21,15))

        ax1 = plt.subplot2grid((7,1), (0,0), rowspan=4)
        plt.scatter(data_A[0][data_A_index],data_A[3][data_A_index],s=6,alpha=0.5,label="KMTA")
        plt.errorbar(data_A[0][data_A_index],data_A[3][data_A_index],yerr=data_A[4][data_A_index],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
        plt.scatter(data_C[0][data_C_index],data_C[3][data_C_index],s=6,alpha=0.5,label="KMTC")
        plt.errorbar(data_C[0][data_C_index],data_C[3][data_C_index],yerr=data_C[4][data_C_index],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
        plt.scatter(data_S[0][data_S_index],data_S[3][data_S_index],s=6,alpha=0.5,label="KMTS")
        plt.errorbar(data_S[0][data_S_index],data_S[3][data_S_index],yerr=data_S[4][data_S_index],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)

        # plt.plot(time_simulate+t0,mag_simulate+19.59,c="black",linewidth=1,label="Best Fit")
        # plt.plot(time_simulate_2+t0,mag_simulate_2+19.59,c="r",linewidth=1,label="MDN Predicted")

        plt.xlabel("t/HJD-2450000",fontsize=16)
        plt.ylabel("Magnitude",fontsize=16)
        plt.legend()
        
        plt.gca().invert_yaxis()

        ax2 = plt.subplot2grid((7,1), (4, 0))
        plt.scatter(data_A[0][data_A_index],data_A[-3][data_A_index],s=6,alpha=0.5,label="KMTA")
        plt.scatter(data_C[0][data_C_index],data_C[-3][data_C_index],s=6,alpha=0.5,label="KMTC")
        plt.scatter(data_S[0][data_S_index],data_S[-3][data_S_index],s=6,alpha=0.5,label="KMTS")
        plt.legend()
        plt.xlabel("t/HJD-2450000")
        plt.ylabel("FWHM")
        ax3 = plt.subplot2grid((7,1), (5, 0))
        plt.scatter(data_A[0][data_A_index],data_A[-2][data_A_index],s=6,alpha=0.5,label="KMTA")
        plt.scatter(data_C[0][data_C_index],data_C[-2][data_C_index],s=6,alpha=0.5,label="KMTC")
        plt.scatter(data_S[0][data_S_index],data_S[-2][data_S_index],s=6,alpha=0.5,label="KMTS")
        plt.legend()
        plt.xlabel("t/HJD-2450000")
        plt.ylabel("Sky")
        ax4 = plt.subplot2grid((7,1), (6, 0))
        plt.scatter(data_A[0][data_A_index],data_A[-1][data_A_index],s=6,alpha=0.5,label="KMTA")
        plt.scatter(data_C[0][data_C_index],data_C[-1][data_C_index],s=6,alpha=0.5,label="KMTC")
        plt.scatter(data_S[0][data_S_index],data_S[-1][data_S_index],s=6,alpha=0.5,label="KMTS")
        plt.legend()
        plt.xlabel("t/HJD-2450000")
        plt.ylabel("secz")
        '''
        '''
        plt.suptitle("KMT-%d-BLG-%04d"%(year,posi,),fontsize=32)

        plt.show()

        plt.savefig("test_realKMT_cut.png")
        plt.close()
    else:
        order = np.argsort(time)
        time = time[order]
        mag = mag[order]
        errorbar = errorbar[order]

        
        plt.figure(figsize=(21,15))

        ax1 = plt.subplot2grid((7,1), (0,0), rowspan=4)
        plt.scatter(data_A[0],data_A[3],s=6,alpha=0.5,label="KMTA")
        plt.errorbar(data_A[0],data_A[3],yerr=data_A[4],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
        plt.scatter(data_C[0],data_C[3],s=6,alpha=0.5,label="KMTC")
        plt.errorbar(data_C[0],data_C[3],yerr=data_C[4],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
        plt.scatter(data_S[0],data_S[3],s=6,alpha=0.5,label="KMTS")
        plt.errorbar(data_S[0],data_S[3],yerr=data_S[4],fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
        if fit:
            plt.plot(time,mag_fit)
            plt.plot(time,mag_fit_mm)

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


