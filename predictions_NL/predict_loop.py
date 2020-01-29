import sys
sys.path.append(r'/home/irene/PycharmProjects/NL_predictors')
import os
import numpy as np
from utils import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

def check_for_extremes(m):
    idx = np.where(np.isnan(m))
    m[idx] = -99
    return m

def tune_name(s):
    template = "NL_Prediction_{0}_{1}_{2}.csv"
    nl, alll, preds, lc, y, m, d = s[:-4].split("_")
    if len(d) == 1:
        day = "0" + d
    else:
        day = d
    if len(m) == 1:
        month = "0" + m
    else:
        month = m
    return template.format(y, month, day)

def get_whiskers(path_train):
    k = 1
    idx = np.genfromtxt(path_train, delimiter=";", usecols=range(0, 4), dtype=str, skip_header=1)
    dic = defaultdict(list)
    for row in idx:
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
        value = float(row[3])
        dic[date.month].append(value)
    data = [dic[i] for i in range(1, 13)]
    ax = plt.boxplot(data)
    dicwhiskers = defaultdict(tuple)
    for i in range(2, 25, 2):
        lower_whisker = ax['whiskers'][i-2].get_ydata()
        upper_whisker = ax['whiskers'][i-1].get_ydata()
        dicwhiskers[k] = [lower_whisker, upper_whisker]
        k += 1
    return dicwhiskers

def bounded_descaling(ypred, dicwhiskers, month):
    # print("Month: ", month)
    low_whisker = dicwhiskers[month][0]
    upp_whisker = dicwhiskers[month][1]
    # print("\t These whiskers: ", low_whisker, upp_whisker)
    low_bound = low_whisker[0] # the lowest of the lower, acts as 'minimum'
    upp_bound = upp_whisker[1] # the uppest of the upper, acts as 'maximum'
    # print("\t These bounds: ", low_bound, upp_bound)
    Y_bounded_desc = np.round(ypred * (upp_bound - low_bound) + low_bound)
    return Y_bounded_desc



################
# Main Program #
################

template = "{0}_{1}"
version = 7
path_train = r"/home/irene/PycharmProjects/NL_predictors/data/random_FS_nymphs_with_zeros_savitzky_golay.csv"
path_testing_tables = r"/home/irene/PycharmProjects/NL_predictors/data/versions/v{0}/testing_tables_v{0}/"
path_pred_rf = r"/home/irene/PycharmProjects/NL_predictors/data/versions/v8/predictions_v8/2014/{0}"

m, all_observations, headers_list, combination, descale, descale_target = load_stuff(path_train, experiment=2)
target_limits = descale[0]

maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))

xtrain = m[:lim, 1:]
ytrain = m[:lim, 0]
xtest = m[lim:, 1:]
ytest = m[lim:, 0]

print(xtrain.shape, ytrain.shape)
print(xtest.shape, ytest.shape)


rf = RandomForestRegressor(n_estimators=500, n_jobs=4, max_features="auto", oob_score=False, bootstrap=True, criterion="mse", max_leaf_nodes=None, max_depth=None, min_samples_leaf=1, min_samples_split=2, warm_start=False)
rf.fit(xtrain, ytrain)

pred_rf = rf.predict(xtest)

pred_rf_descaled = np.round(descaling(pred_rf, target_limits), decimals=0)
ytest_desc = np.round(descaling(ytest, target_limits), decimals=0)
pack = show_model_evaluation(ytest_desc, pred_rf_descaled)
print(pack)

dicwhiskers = get_whiskers(path_train)

for subdir, dirs, files in os.walk(path_testing_tables.format(version)):
    for file in files:
        print("\n\n")
        print("-" * 80)
        print("Testing file:  \t", file)
        print("-" * 80)
        name = tune_name(file)

        c, b, year, month, day = name[:-4].split("_")

        location_test_file = path_testing_tables.format(version)+ "/" + file
        xraw = np.loadtxt(location_test_file, skiprows=1, delimiter=";", usecols=range(3, 89))
        idx = np.loadtxt(location_test_file, skiprows=1, delimiter=";", usecols=range(0, 3))

        xtest = scale_NL(xraw)
        xtest[np.isnan(xtest)] = 0
        pred_rf = rf.predict(xtest).reshape(-1, 1)
        # pred_rf_descaled = descaling(pred_rf, target_limits).reshape(-1, 1)
        pred_rf_bounded_desc = bounded_descaling(pred_rf, dicwhiskers, int(float(month)))
        out_rf = np.hstack((idx, pred_rf_bounded_desc))
        np.savetxt(path_pred_rf.format(name), out_rf, delimiter=";")
