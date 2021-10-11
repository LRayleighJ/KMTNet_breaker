import os
import numpy as np

targetdir = "/mnt/e/KMT_catalog/"

years = ['2016','2017','2018','2019',"2020"]
num_events = [2588, 2817, 2781, 3303, 894]

for kk in range(len(years)):
    for i in np.arange(num_events[kk]):
        i_blyat = i+1
        file_number = "%04d" % i_blyat
        folder_name = years[kk]+ '_' + file_number
        file_name = 'http://kmtnet.kasi.re.kr/~ulens/event/'+years[kk]+'/data/KB'+years[kk][2:]+str(file_number)+'/pysis/pysis.tar.gz'
        os.system("mkdir " + targetdir + folder_name)
        os.system("wget -O "+ targetdir + "temp.tar.gz " +file_name)
        os.system("tar -xvzf "+targetdir+"temp.tar.gz -C "+ targetdir + folder_name)
        os.system("rm -f "+targetdir+"temp.tar.gz")
        print(folder_name," has completed")