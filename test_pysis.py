import numpy as np

def killstr(data):
    return data[~(data == "___").any(axis=1)].astype(np.float)

x = np.loadtxt("test.pysis",dtype=str)
print(x)
print(~(x == "___").any(axis=1))
print(killstr(x))
