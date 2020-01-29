import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import numpy as np
from utils import *
from sklearn.model_selection import GridSearchCV
from  time import time
from sklearn.svm import SVR
import matplotlib.pyplot as plt

def report(results, n_top=20):
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

print("\nSUPPORT VECTOR REGRESSION CALIBRATION")
print("-" * 50, "\n")
print("Training observation samples: ", xtrain.shape)
print("Training target samples: ", ytrain.shape)

param_grid_v1 = {"C": [0.15, 0.25, 0.50, 0.75, 0.95, 1, 1.5, 2],
              "gamma": [0.01, 0.05, 0.1, 0.25, 0.5],
              "epsilon": [0.1, 0.25, 0.5, 0.75, 1],
              "shrinking": [True, False],
              "tol": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
              }

param_grid_v2 = {"C": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
              "gamma": [0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.03, 0.05],
              "epsilon": [0.01, 0.02, 0.03, 0.04, 0.05, 0.1],
              "shrinking": [True, False],
              "tol": [0.0005, 0.001, 0.003, 0.005]
              }

param_grid = {"C": [4, 4.5, 5, 5.5, 6, 6.5, 7],
              "gamma": [0.009, 0.0095, 0.01, 0.015],
              "epsilon": [0.01, 0.015, 0.02, 0.025],
              "shrinking": [True, False],
              "tol": [0.0005, 0.0007, 0.001, 0.0015, 0.002, 0.003, 0.005]
              }



# svr = SVR(kernel='rbf')
# print("Starting...")
# # run grid search
# grid_search = GridSearchCV(svr, param_grid=param_grid)
# start = time()
# grid_search.fit(xtrain, ytrain)
#
# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))
# report(grid_search.cv_results_)


svr = SVR(kernel='rbf', shrinking=True, tol=0.001, gamma=0.002, epsilon=0.015, C=4)
svr.fit(xtrain, ytrain)
pred_svr = svr.predict(xtest)
pred_svr_desc = np.round(descaling(pred_svr, target_limits), decimals=0)
ytest_desc = np.round(descaling(ytest, target_limits), decimals=0)
rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_desc, pred_svr_desc)
print(rmse, rmses, rmseu, nrmse, mae, r2)

plt.plot(pred_svr_desc, ytest_desc, "o")
plt.plot(pred_svr_desc, pred_svr_desc, "-")
plt.show()