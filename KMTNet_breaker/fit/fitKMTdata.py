import MulensModel as mm
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import os
import re
import pandas as pd
import KMTNet_breaker.loaddata.loadKMTdata as loaddata
import scipy.optimize as op
from scipy.optimize import curve_fit



def mag_cal(t,t0,tE,u0,m0):
    u = np.sqrt(((t-t0)/tE)**2+u0**2)
    A = (u**2+2)/(u*np.sqrt(u**2+4))
    return m0-2.5*np.log10(A)

def chi_square(single,binary,sigma):
    return np.sum(np.power((single-binary)/sigma,2))

def chi2_for_minimize(args,time,mag,sigma):
    return chi_square(mag_cal(time,*args),mag,sigma)


def chi2_for_model(theta, event, parameters_to_fit):
    """for given event set attributes from parameters_to_fit
    (list of str) to values from the theta list"""
    for (key, parameter) in enumerate(parameters_to_fit):
        setattr(event.model.parameters, parameter, theta[key])
    return event.get_chi2()

'''
args,data,_,_,_= loaddata.getKMTdata(year=2018,posi=1046,cut=1,cutratio=2)
print(args)

'''

def fit_single(time,mag,errorbar,initial_guess):

    t_0_KMT = initial_guess[-4]
    t_E_KMT = initial_guess[-3]
    u_0_KMT = initial_guess[-2]
    Ibase_KMT = initial_guess[-1]# np.mean(np.sort(mag)[-50:])
    '''
    pspl_model = mm.Model({'t_0': t_0_KMT, 'u_0': u_0_KMT, 't_E': t_E_KMT})
    pspl_model.set_datasets([mydataset])

    my_event = mm.Event(datasets=mydataset, model=pspl_model)
    chi2_initial = my_event.get_chi2()
    print(my_event.model.parameters)
    print("give chi^2 of {:.2f}.".format(chi2_initial))

    parameters_to_fit = ["t_0", "u_0", "t_E"]
    initial_guess = [t_0_KMT, u_0_KMT, t_E_KMT]

    result = op.minimize(chi2_for_model, x0=initial_guess,args=(my_event, parameters_to_fit), method='Nelder-Mead')


    print("Fitting was successful? {:}".format(result.success))
    if not result.success:
        print(result.message)
    print("Function evaluations: {:}".format(result.nfev))
    if isinstance(result.fun, np.ndarray):
        if result.fun.ndim == 0:
            result_fun = float(result.fun)
        else:
            result_fun = result.fun[0]
    else:
        result_fun = result.fun
    print("The smallest function value: {:.3f}".format(result_fun))
    print("for parameters: {:.5f} {:.4f} {:.3f}".format(*result.x.tolist()))

    
    


    left_bound = [0.5*t_E_KMT,t_0_KMT-0.5*t_E_KMT,u_0_KMT-1,0.7*Ibase_KMT]
    right_bound = [1.5*t_E_KMT,t_0_KMT+0.5*t_E_KMT,u_0_KMT+1,1.3*Ibase_KMT]
    
    try:
        popt, pcov = curve_fit(mag_cal,time,mag,sigma=errorbar,bounds=(left_bound,right_bound))
    except:
        print("Curvefit error")
        continue
    lc_fit = mag_cal(time,*popt)

    chi_s = chi_square(lc_fit,mag,errorbar)
    print(popt)
    print("chi^2 for curvefit: ", chi_s)

    '''
    
    # minimize

    # initial_guess = [t_0_KMT,t_E_KMT,u_0_KMT,Ibase_KMT]


    result = op.minimize(chi2_for_minimize, x0=initial_guess,args=(time,mag,errorbar), method='Nelder-Mead')
    

    # print("Fitting was successful? {:}".format(result.success))
    if not result.success:
        print(result.message)
    # print("Function evaluations: {:}".format(result.nfev))
    if isinstance(result.fun, np.ndarray):
        if result.fun.ndim == 0:
            result_fun = float(result.fun)
        else:
            result_fun = result.fun[0]
    else:
        result_fun = result.fun
    # print(result.x.tolist())
    args_minimize = result.x.tolist()
    lc_fit_minimize = mag_cal(time,*args_minimize)

    chi_s_minimize = chi_square(lc_fit_minimize,mag,errorbar)
    # print("chi^2 for minimize: ",chi_s_minimize)

    return result.x.tolist(), chi_s_minimize

    """
    popt_2, pcov_2 = curve_fit(mag_cal,time,mag,bounds=(left_bound,right_bound))
    lc_fit_2 = mag_cal(time,*popt_2)

    chi_s_2 = chi_square(lc_fit_2,mag,errorbar)
    print(popt_2)
    print(chi_s_2)
    

    plt.figure(figsize=[20,12])
    
    plt.errorbar(time,mag,yerr=errorbar,fmt='o',capsize=2,elinewidth=1,ms=0,zorder=0, alpha=0.5)
    plt.plot(time,lc_fit,label = "curvefit")
    plt.plot(time,lc_fit_minimize,label = "minimize")
    plt.xlabel("t/HJD-2450000",fontsize=16)
    plt.ylabel("Magnitude",fontsize=16)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.title("$\chi^2$(curvefit): %.3f $\chi^2$(minimize): %.3f label: %d "%(chi_s,chi_s_minimize,int(labels[-1])))


    plt.savefig("./fig_simu_2/testfit_simu_%d.png"%(index,))
    plt.close()
    correct += 1


print(correct)
"""