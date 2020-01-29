import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
from utils import *
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import csv

################
# Main program #
################

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_savitzky_golay.csv"
path_test = r"D:\GeoData\workspaceimg\Special\IJGIS\testing_tables_v4\NL_All_Predictors_LC_1_8_2014.csv"
path_pred_rf = r"D:\GeoData\workspaceimg\Special\IJGIS\predictions_v4\RF_NL_All_Predictors_LC_1_8_2014.csv"

m, all_observations, headers_list, combination, descale, descale_target = load_stuff(path_in, experiment=2)

target_limits = descale[0]
maxlim = m.shape[0]

lim = int(np.divide(maxlim * 70, 100))

# Read and prepare data
xtrain = m[:lim, 1:]
ytrain = m[:lim, 0]
xtest = np.loadtxt(path_test,  delimiter=";", skiprows=1, usecols=range(3, 89))
idx = np.loadtxt(path_test,  delimiter=";", skiprows=1, usecols=range(0, 3))


print("\nxtrain: {0} \t\t ytrain: {1}".format(xtrain.shape, ytrain.shape))
print("\nRANDOM FOREST MODELLING")
print("-" * 30, "\n")
print("Training observation samples: ", xtrain.shape)
print("Training target samples: ", ytrain.shape)
print("Starting...")

rf = RandomForestRegressor(n_estimators=500, n_jobs=4, max_features="auto", oob_score=False, bootstrap=True, criterion="mse", max_leaf_nodes=None, max_depth=None, min_samples_leaf=1, min_samples_split=1, warm_start=False)
rf.fit(xtrain, ytrain)
pred_rf = np.round(rf.predict(xtest), decimals=5)
pred_rf_desc = np.round(descaling(pred_rf, target_limits), decimals=0).reshape(-1, 1)

print(np.unique(pred_rf))
print(np.unique(pred_rf).shape)
print(np.unique(pred_rf_desc))

print(pred_rf_desc.shape)
print(idx.shape)

out_rf = np.hstack((idx, pred_rf_desc))
np.savetxt(path_pred_rf, out_rf, delimiter=";")
