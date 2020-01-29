import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils import *
from evaluate import evaluate
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
from scipy.stats import skew, kurtosis
import datetime
import scipy.stats
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from collections import defaultdict

def embed_metrics(rmse, rmses, rmseu, nrmse):
    return '$\mathrm{RMSE}=%.2f$\n$\mathrm{RMSEs}=%.2f$\n$\mathrm{RMSEu}=%.2f$\n$\mathrm{NRMSE}=%.2f$'%(rmse, rmses, rmseu, nrmse)

def stamp(name, rmse, rmses, rmseu, nrmse, mae, r2):
    print("\n {0}".format(name))
    print("-"*30, "\n")
    print("RMSE (ytest and y_pred): ", rmse)
    print("\tRMSE systematic: ", rmses)
    print("\tRMSE UNsystematic: ", rmseu)
    print("\tNRMSE: ", nrmse)
    print("\nMAE (ytest and y_pred):", mae)
    print("R2 coefficient (ytest and y_pred: ", r2)

def plot_model(name, predictions, test, labels, number):
    plt.subplot(2, 1, number)
    plt.title(tmp.format(name,len(ytest_desc)), size=14)
    plt.plot(predictions, test, 'o', label=u'Observations', color="#7F7E7A")
    # Add textbox summarizing the metrics used
    plt.annotate(textstr, xy=(1, 0), xycoords='axes fraction', fontsize=14,
        xytext=(-10, 10), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='gray', alpha=0.3),)

    # for label, x, y in zip(labels, predictions, test):
    #     # print(label, x, y)
    #     plt.annotate(label, xy = (x, y))

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    stdev = np.round(np.std(test), decimals=2)
    xlinspace = np.linspace(0, 599, 600)
    plt.plot(predictions, test, "o")
    plt.plot(xlinspace, xlinspace, "-", color="black")
    # plt.plot(xlinspace+15, xlinspace, "-", color="gray")
    # plt.plot(xlinspace-15, xlinspace, "-", color="gray")
    plt.fill_between(xlinspace, xlinspace-stdev, xlinspace+stdev, color='grey', alpha='0.15')
    plt.grid()

def show_FI(ensemble, combination, headers, lim):
    # print("Len of headers: ", len(headers))
    # print("Len of combi: ", len(combination))
    # print("Combination: ", combination)
    # cols = [headers[item] for item in combination[1:]] # Skip label tick count
    cols = headers[1:]
    i = 1
    print("\nFeature Importances")
    print("-"*40)
    for item in list(reversed(sorted(zip(cols, ensemble.feature_importances_), key=itemgetter(1))))[:lim]:
        print(i, ")\t", item[0], "\t\t", np.round(item[1] * 100, decimals=2), "%")
        i += 1


def general_checks(ytest, ytest_desc, pred_rf, pred_rf_desc, pred_gbr, pred_gbr_desc):
    print("-" * 30, "\n")
    print("Scaled and descaled means of target, RF and GBR")
    print("\ttarget: {0} / {1}".format(round(np.mean(ytest), 3), round(np.mean(ytest_desc), 3)))
    print("\tRF: {0} / {1}".format(round(np.mean(pred_rf), 3), round(np.mean(pred_rf_desc), 3)))
    print("\tGBR: {0} / {1}".format(round(np.mean(pred_gbr), 3), round(np.mean(pred_gbr_desc), 3)))
    print("")
    print("Scaled and descaled stdev of target, RF and GBR")
    print("\ttarget: {0} / {1}".format(round(np.std(ytest), 3), round(np.std(ytest_desc), 3)))
    print("\tRF: {0} / {1}".format(round(np.std(pred_rf), 3), round(np.std(pred_rf_desc), 3)))
    print("\tGBR: {0} / {1}".format(round(np.std(pred_gbr), 3), round(np.std(pred_gbr_desc), 3)))
    print("")
    print("First 20 values of target, RF and GBR")
    print("\ttarget: ", ytest_desc[:20].tolist())
    print("\tRF: \t", pred_rf_desc[:20].tolist())
    print("\tGBR: \t", pred_gbr_desc[:20].tolist())
    print("")


def add_labels(ax, obs, lim, predictions, truevalues):
    l_labels = []
    for item in obs[lim:]:
        l_labels.append(item[1])

    for label, x, y in zip(l_labels, predictions, truevalues):
        if y>60:
            plt.annotate(label, xy = (x+2, y+3))

def select_columns(m, combination):
    return m[:, combination]

def encode_labels(m):
    combination = tuple(range(1, 16)) + tuple(range(99, 102))
    df = pd.DataFrame(select_columns(m, combination))
    dummies = pd.get_dummies(df, columns=list(range(0, 18))).as_matrix()
    return dummies

def prepare_labels(observations):
    labels = []
    for obs in observations:
        d = datetime.datetime.strptime(obs[2], "%Y-%m-%d")
        n = obs[1][0:3]
        l = n + "_" + str(d.year)[2:] + "_" + str(d.month)
        labels.append(l)
    return labels

def target_descaling(dicdesc, all_obs, ypred):
    descaled = []
    i = 0
    for observation in all_obs:
        name = observation[1]
        mean, std = dicdesc[name]
        value = ypred[i]
        t_descaled = np.multiply(value, std) + mean
        descaled.append(t_descaled)
    return np.array(descaled)

def masking(mask, obs):
    l = []
    dic = defaultdict(int)
    for row in mask:
        dic[row[0]] = row[1]
    for row in obs:
        key = int(row[0])
        l.append(dic[key])
    return l


################
# Main Program #
################

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_savitzky_golay.csv"
tmp = "{0}: predictions vs test ({1} samples)"

path_mask = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_mask.csv"


m, all_observations, headers_list, combination, descale, descale_target = load_stuff(path_in, experiment=0)
print(all_observations[0])
# headers = headers_list[0:102] # + ["doy"]# + ["s_tmin","i_tmin","s_tmax","i_tmax","s_prec","i_prec","s_ev","i_ev","s_rh","i_rh"]

banned = ["Was_2_13_5", "Was_1_13_5", "Gie_2_11_5", "Mon_1_10_7", "Was_1_12_5", "Nij_1_07_4", "Gie_2_11_10", "Ede_2_08_8", "App_1_11_5", "App_1_09_3", "Gie_1_12_11"]
labels = prepare_labels(all_observations)

headers = headers_list[:]

target_limits = descale[0]
maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))
mask = np.loadtxt(path_mask, delimiter=";")

algo = masking(mask, all_observations[lim:])
print("algo: ", len(algo))
print("zeros: ", algo.count(0))

# Read and prepare data
xtrain = m[:lim, 1:]
ytrain = m[:lim, 0]
xtest = m[lim:, 1:]
ytest = m[lim:, 0]

print("\nxtrain: {0} \t\t ytrain: {1} \t\t xtest: {2} \t\t ytest: {3}".format(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape))
print("\nRANDOM FOREST MODELLING")
print("-" * 30, "\n")
print("Training observation samples: ", xtrain.shape)
print("Training target samples: ", ytrain.shape)
print("Starting...")

rf = RandomForestRegressor(n_estimators=250,
                           n_jobs=4,
                           max_features="auto",
                           oob_score=False,
                           bootstrap=True,
                           criterion="mse",
                           max_leaf_nodes=None, # This goes in pairs with max_depth
                           max_depth=None, # and None means maximum development of trees
                           min_samples_leaf=1,
                           min_samples_split=1,
                           warm_start=False)
print("Created ensemble...")
rf.fit(xtrain, ytrain)
pred_rf = rf.predict(xtest)

print("\nGRADIENT BOOSTING REGRESSION MODELLING")
print("-" * 30, "\n")
print("Training observation samples: ", xtrain.shape)
print("Training target samples: ", ytrain.shape)
print("Starting...")
gbr = GradientBoostingRegressor(loss='huber',
                                alpha=0.95,
                                learning_rate=.05,
                                max_features=None,
                                max_leaf_nodes=None,
                                criterion="mse",
                                min_samples_split=2,
                                min_samples_leaf=3,
                                n_estimators=400,
                                max_depth=3)
print("Training...\n")
gbr.fit(xtrain, ytrain)

# # compute test set deviance
# test_deviance = np.zeros((gbr.get_params()['n_estimators'],), dtype=np.float64)
#
# for i, y_pred in enumerate(gbr.staged_decision_function(xtest)):
#     # clf.loss_ assumes that y_test[i] in {0, 1}
#     test_deviance[i] = gbr.loss_(ytest, y_pred)
#
# plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5], '-', linewidth=3, color="blue", label="GBR Deviance")
# plt.grid()
# plt.legend(loc='upper left')
# plt.xlabel('Boosting Iterations')
# plt.ylabel('Test Set Deviance')
# plt.show()

print("Predicting")
pred_gbr = gbr.predict(xtest)

print("Descaling predictions and target")
pred_rf_desc = np.round(descaling(pred_rf, target_limits), decimals=0)
pred_gbr_desc = np.round(descaling(pred_gbr, target_limits), decimals=0)
ytest_desc = np.round(descaling(ytest, target_limits), decimals=0)

# pred_rf_desc = target_descaling(descale_target, all_observations[lim:], pred_rf)
# pred_gbr_desc = target_descaling(descale_target, all_observations[lim:], pred_gbr)
# ytest_desc = target_descaling(descale_target, all_observations[lim:], ytest)

plt.suptitle("Histogram and skewness for target, GBR and RF. Skewness: > 0 means that there is + weight in the left tail of the dist.")

# plt.subplot(1, 3, 1)
# skewness = skew(ytest_desc)
# plt.title("Target for training (Skewness: {0})".format(skewness))
# plt.hist(ytest_desc, bins=90)
# # plt.hist(np.sqrt(ytest_desc), bins=30)
# plt.xlim(0, 90)
# plt.ylim(0, 180)
#
# plt.subplot(1, 3, 2)
# skewness = skew(pred_gbr_desc)
# plt.title("GBR prediction (Skewness: {0})".format(skewness))
# plt.hist(pred_gbr_desc, bins=90)
# plt.xlim(0, 90)
# plt.ylim(0, 180)
#
# plt.subplot(1, 3, 3)
# skewness = skew(pred_rf_desc)
# plt.title("RF prediction (Skewness: {0})".format(skewness))
# plt.hist(pred_rf_desc, bins=90)
# plt.xlim(0, 90)
# plt.ylim(0, 180)
# plt.show()

 # Order: rmse, rmses, rmseu, nrmse, mae, r2, the_mean, the_mean_pred


keep1 = []
keep2 = []
keep3 = []
for i in range(len(ytest_desc)):
    if algo[i] == 1:
        keep1.append(ytest_desc[i])
        keep2.append(pred_gbr_desc[i])
        keep3.append(pred_rf_desc[i])

ytest_desc = np.array(keep1)
pred_gbr_desc = np.array(keep2)
pred_rf_desc = np.array(keep3)




evaluate_gbr = show_model_evaluation(ytest_desc, pred_gbr_desc)
general_checks(ytest, ytest_desc, pred_rf, pred_rf_desc, pred_gbr, pred_gbr_desc)

print("Metrics and indicators: ")
rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_desc, pred_rf_desc)
textstr = embed_metrics(rmse, rmses, rmseu, nrmse)
stamp("Random Forest", rmse, rmses, rmseu, nrmse, mae, r2)
plot_model("RF", pred_rf_desc, ytest_desc, labels[lim:], 1)
show_FI(rf, combination, headers, 20)
# add_labels(ax1, all_observations, lim, pred_rf_desc, ytest_desc)

rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_desc, pred_gbr_desc)
textstr = embed_metrics(rmse, rmses, rmseu, nrmse)
stamp("Gradient Boosting", rmse, rmses, rmseu, nrmse, mae, r2)
plot_model("GBR", pred_gbr_desc, ytest_desc, labels[lim:], 2)
show_FI(gbr, combination, headers, 20)
# add_labels(ax2, all_observations, lim, pred_gbr_desc, ytest_desc)

plt.show()



# params = {"loss":"huber",
#           'learning_rate': 0.05,
#           'max_features': 30,
#           'max_leaf_nodes': None,
#           'criterion': "mse",
#           'min_samples_split': 2,
#           "min_samples_leaf":3,
#           "n_estimators":250,
#           "max_depth":5}
#
# test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
#
# for i, y_pred in enumerate(gbr.staged_predict(xtest)):
#     test_score[i] = gbr.loss_(ytest, pred_gbr)
#
# plt.figure(figsize=(12, 6))
# plt.subplot(1, 2, 1)
# plt.title('Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, gbr.train_score_, 'b-',
#          label='Training Set Deviance')
# plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
#          label='Test Set Deviance')
# plt.legend(loc='upper right')
# plt.xlabel('Boosting Iterations')
# plt.ylabel('Deviance')
# plt.show()






