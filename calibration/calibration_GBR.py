import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
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
ytrain = m[:lim, 0]
xtest = m[lim:, 1:]
ytest = m[lim:, 0]

print("\nxtrain: {0} \t\t ytrain: {1} \t\t xtest: {2} \t\t ytest: {3}".format(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape))

print("\nGRADIENT BOOSTING REGRESSION CALLIBRATION")
print("-" * 30, "\n")
print("Training observation samples: ", xtrain.shape)
print("Training target samples: ", ytrain.shape)
print("Starting...")

param_grid = {"loss": ["ls", "lad", "huber", "quantile"],
              "learning_rate": [0.2, 0.1, 0.05, 0.01],
              "max_depth": [1, 3, 5, 10, 20],
              "criterion": ["friedman_mse", "mse"],
              "min_samples_split": [2, 5, 9],
              "min_samples_leaf": [1, 3],
              "max_features":[1, 3, 10, 30, "auto", "sqrt", "log2"],
              "max_leaf_nodes":[None],
              "alpha":[0.95]
            }

gbr = GradientBoostingRegressor(n_estimators=250)

# run grid search
grid_search = GridSearchCV(gbr, param_grid=param_grid)
start = time()
grid_search.fit(xtrain, ytrain)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)
