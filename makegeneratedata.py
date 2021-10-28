import numpy as np
import multiprocessing as mp
import random
import KMTNet_breaker.loaddata.loadKMTdata as loaddata
import matplotlib.pyplot as plt

# getnoisehist

years_list = [2017, 2018,2019,2020]
num_events = [2817, 2781, 3303, 894]

years_seq = (years_list[0]*np.ones(num_events[0])).astype(np.int)
num_seq = np.arange(num_events[0])
filelist = np.array([years_seq,num_seq]).T


for i in range(1,len(years_list)):
    years_seq = (years_list[i]*np.ones(num_events[i])).astype(np.int)
    num_seq = np.arange(num_events[i])
    year_combine = np.array([years_seq,num_seq]).T
    filelist = np.append(filelist,year_combine,axis=0)

# 9795

# generate noisemodel
size_batch = 100
mags = []
sigmas = []

for i in range(size_batch):
    try:
        index_year_posi = filelist[random.randint(0,9794)]

        args,data,_,_,_= loaddata.getKMTdata(year=index_year_posi[0],posi=index_year_posi[1],cut=0)

        mags = np.append(mags,data[1])
        sigmas = np.append(sigmas,data[2])
        print(i)
    except Exception as e:
        print(e.__class__.__name__,e)
        print(index_year_posi)
        continue

plt.figure(figsize=(12,18))
plt.subplot(311)
plt.hist(mags,bins=500)
plt.subplot(312)
plt.hist(sigmas,bins=500)
plt.subplot(313)
plt.scatter(mags,sigmas,s=1)
plt.savefig("testmaghist.png")
'''
def generateRandseed(x):
    return np.random.seed(x*random.randint(0,1+x*10))

def test(x):
    np.random.RandomState()
    np.random.seed(x*random.randint(0,1+x*10))
    print("test:",x,np.random.randint(0,100))

if __name__=="__main__":
    with mp.Pool() as p:
        p.map(test, range(50))
'''
