input = "/media/dani/data/ncaltech_experiments/N-Caltech101/"
output = "/media/dani/data/ncaltech_experiments/N-Caltech101_numpy/"

import glob
import os
import pandas as pd
import numpy as np
from multiprocessing import Pool

def convert(f):
    substring = os.path.join(*f.split("/")[-2:])
    substring = substring[:-3] + "npy"
    #print(os.path.join(output, substring))
    data = pd.read_pickle(f).as_matrix(["x", "y", "time", "polarity"]).astype(np.float32)
    #print(data[:10])
    np.save(os.path.join(output, substring), data)

folders = [os.path.join(output, f) for f in os.listdir(input)]
for f in folders:
    if not os.path.isdir(f):
        os.mkdir(f)

files = glob.glob(os.path.join(input, "*", "*.pkl"))
pool = Pool(4)
pool.map_async(convert, files).get(99999999)