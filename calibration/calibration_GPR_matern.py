import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from utils import *
from sklearn.model_selection import GridSearchCV
from  time import time
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared, Matern
from sklearn.gaussian_process import GaussianProcessRegressor

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


################
# Main Program #
################

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_savitzky_golay.csv"

m, all_observations, headers_list, combination, descale = load_stuff(path_in, experiment=0)
target_limits = descale[0]

maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))

# Read and prepare data
xtrain = m[:lim, 1:]
ytrain = m[:lim, 0]
xtest = m[lim:, 1:]
ytest = m[lim:, 0]

print("\nxtrain: {0} \t\t ytrain: {1} \t\t xtest: {2} \t\t ytest: {3}".format(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape))

print("\nGAUSSIAN PROCESS CALIBRATION (Matern kernel)")
print("-" * 40, "\n")
print("Training observation samples: ", xtrain.shape)
print("Training target samples: ", ytrain.shape)
print("Starting...")

param_grid = {"n_restarts_optimizer": [3],
              "alpha": [0.25, 0.5, 0.75, 1]
            }

list_length_scale = np.arange(0, 1, 0.05)
list_length_scale_bounds = [(1e-1, 1.0), (1e-1, 5.0)]
list_nu = [1.5, 2.5]

for l in list_length_scale:
    for b in list_length_scale_bounds:
        for n in list_nu:
            print("")
            print("GPR Matern with parameters: ", l, b, n)
            gp_kernel = 1.0 * Matern(length_scale=l, length_scale_bounds=b, nu=n)
            gpr = GaussianProcessRegressor(kernel=gp_kernel)
            # run grid search
            grid_search = GridSearchCV(gpr, param_grid=param_grid)
            start = time()
            grid_search.fit(xtrain, ytrain)
            print("\tGridSearchCV took %.2f seconds for %d candidate parameter settings."
                  % (time() - start, len(grid_search.cv_results_['params'])))
            report(grid_search.cv_results_)




