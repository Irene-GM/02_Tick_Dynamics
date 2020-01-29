import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from utils import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict

def descaling(target, descale):
    minimum = 0
    maximum = descale[2]
    X_scaled = np.round(target * (maximum - minimum) + minimum)
    return X_scaled


def heldout_score(clf, X_test, y_test):
    """compute deviance scores on ``X_test`` and ``y_test``. """
    n_estimators = clf.get_params()["n_estimators"]
    score = np.zeros((n_estimators,), dtype=np.float64)
    for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
        score[i] = clf.loss_(y_test, y_pred)
    return score


################
# Main program #
################


path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_savitzky_golay.csv"
tmp = "{0}: predictions vs test ({1} samples)"

m, all_observations, headers_list, combination, descale = load_stuff(path_in, experiment=0)
headers = headers_list[0:102] + ["yminusone"]#, "yminustwo"] # Like this because there is a weird symbol
target_limits = descale[0]
maxlim = m.shape[0]

# Read and prepare data
xtrain_oob = m[:, 1:]
ytrain_oob = m[:, 0]

gbr = GradientBoostingRegressor(loss='huber',
                                alpha=0.95,
                                learning_rate=.05,
                                max_features=None,
                                max_leaf_nodes=None,
                                criterion="mse",
                                min_samples_split=2,
                                min_samples_leaf=3,
                                n_estimators=1000,
                                presort=False,
                                subsample=0.2,
                                max_depth=5)

array_estimators = np.arange(gbr.get_params()["n_estimators"]) + 1

print("\nGRADIENT BOOSTING REGRESSION MODELLING")
print("-" * 30, "\n")
print("Training observation samples: ", xtrain_oob.shape)
print("Training target samples: ", ytrain_oob.shape)
print("Training...\n")

# gbr.fit(xtrain_oob, ytrain_oob)
# the_splits = 10
# cv = KFold(n_splits=the_splits)
# val_scores = np.zeros((gbr.get_params()["n_estimators"],), dtype=np.float64)
#
# for train, test in cv.split(xtrain_oob, ytrain_oob):
#     gbr.fit(xtrain_oob[train], ytrain_oob[train])
#     val_scores += heldout_score(gbr, xtrain_oob[test], ytrain_oob[test])
# val_scores /= the_splits
#
# cv_score = val_scores[:]
# cv_score -= cv_score[0]
# cv_best_iter = array_estimators[np.argmin(cv_score)]
#
# plt.plot(array_estimators, cv_score, label='CV loss', color="blue", linewidth=2)
# plt.axvline(x=cv_best_iter, color="red", linewidth=3)
# plt.show()

pred_gbr = cross_val_predict(gbr, xtrain_oob, ytrain_oob, cv=5)
pred_gbr_desc = np.round(descaling(pred_gbr, target_limits), decimals=0)
ytest_desc = np.round(descaling(ytrain_oob, target_limits), decimals=0)

xlinspace = np.linspace(0, len(pred_gbr_desc)-1, len(pred_gbr_desc))
plt.plot(xlinspace, pred_gbr_desc, "-", color="red")
plt.plot(xlinspace, ytest_desc, "-", color="blue")
plt.show()

evaluate_gbr = show_model_evaluation(ytest_desc, pred_gbr_desc)
print(evaluate_gbr)