import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels \
import RBF

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_raw.csv"

m, all_observations, headers_list, combination, descale = load_stuff(path_in, experiment=0)

maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))

idx = np.array(list(range(0, maxlim))).reshape(-1, 1)
print(idx.shape)
target = m[:,0]
print(target.shape)



gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0,
                              optimizer=None, normalize_y=True)
gp.fit(idx, target)



