# KMTNet_breaker

These codes are used for dealing with the data of KMTNet.

* `download.py`: Downloading the data of KMTNet using `multiprocessing`, saving in `rootdir`. But this code is not ready for use. 
* `download_ori.py`: Downloading the data of KMTNet. You can create the KMTNet catalog on our platform, but this code is simple-processing so it will be a little slow.
* `drawKMTdata.py`: Visualizing data from KMTNet.
* `getKMTURLargs.py`: The website of KMTNet provides the simple-lensing fitting arguments (mainly the closest time $t_0$, impact parameter `u_0` and Einstein crossing time $t_E$. Sometimes you will see $t_{eff}$, it means $t_eff = u_0t_E$). Run this code, and the args will be stored in `KMT_args.npy`

To build your own catalog in your PC, please first run `download.py`