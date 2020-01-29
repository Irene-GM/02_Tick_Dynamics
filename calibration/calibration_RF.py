import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils import *
from sklearn.model_selection import GridSearchCV
from  time import time
from evaluate import evaluate
import matplotlib.pyplot as plt

def report(results, n_top=15):
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
ytrain = np.sqrt(m[:lim, 0])
xtest = m[lim:, 1:]
ytest = np.sqrt(m[lim:, 0])

print("\nxtrain: {0} \t\t ytrain: {1} \t\t xtest: {2} \t\t ytest: {3}".format(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape))

print("\nRANDOM FOREST CALIBRATION")
print("-" * 30, "\n")
print("Training observation samples: ", xtrain.shape)
print("Training target samples: ", ytrain.shape)

param_grid = {"max_depth": [3, 5, 10, 15, 20, 25, None],
              "max_features": [1, 3, 10, 30, "auto", "sqrt", "log2"],
              "min_samples_split": [1, 3, 5, 10],
              "min_samples_leaf": [1, 3, 5, 10],
              "bootstrap": [True],
              "max_leaf_nodes": [None],
              "oob_score":[False],
              "criterion": ["mse"]}


print("Starting...")
rf = RandomForestRegressor(n_estimators=300)
print("Created ensemble...")

# run grid search
grid_search = GridSearchCV(rf, param_grid=param_grid)
start = time()
grid_search.fit(xtrain, ytrain)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)



