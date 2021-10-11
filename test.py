import numpy as np
import pandas as pd

data = pd.DataFrame(data=np.load("KMT_args_test.npy",allow_pickle=True))
print(np.load("KMT_args.npy",allow_pickle=True))

print(data.loc[data["index"]=="2020_0010"])
