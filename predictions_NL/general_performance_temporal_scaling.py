import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import os
import csv
from operator import itemgetter
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils import *
from collections import defaultdict
import matplotlib.pyplot as plt
from operator import itemgetter
import pandas as pd
import datetime
import seaborn as sb
import matplotlib as mpl
from sklearn.metrics import r2_score
from matplotlib.dates import DateFormatter

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

def plot_the_model(name, predictions, test, labels, number):
    print("This is the name ", name)
    if name == "gbr":
        name = "gb"
    plt.subplot(2, 1, number)
    plt.subplots_adjust(hspace=.5)
    plt.title(tmp.format(name.upper(),len(ytest_desc)), size=14)
    plt.plot(predictions, test, 'o', label=u'Observations', color="#636769", markeredgewidth=1.0, markeredgecolor='black')
    # Add textbox summarizing the metrics used
    plt.annotate(textstr, xy=(1, 0), xycoords='axes fraction', fontsize=14,
        xytext=(-10, 10), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='grey', alpha=0.3),)

    # for label, x, y in zip(labels, predictions, test):
    #     # print(label, x, y)
    #     plt.annotate(label, xy = (x, y))

    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.xlabel("Predictions")
    plt.ylabel("True values")
    stdev = np.round(np.std(test), decimals=2)
    xlinspace = np.linspace(0, 599, 600)
    plt.plot(xlinspace, xlinspace, "-", color="black", linewidth=2)
    # plt.plot(xlinspace+15, xlinspace, "-", color="gray")
    # plt.plot(xlinspace-15, xlinspace, "-", color="gray")
    plt.fill_between(xlinspace, xlinspace-stdev, xlinspace+stdev, color='grey', alpha='0.30')
    plt.grid()

def show_FI(ensemble, combination, headers, lim):
    # print("Len of headers: ", len(headers))
    # print("Len of combi: ", len(combination))
    # print("Combination: ", combination)
    # cols = [headers[item] for item in combination[1:]] # Skip label tick count
    # cols = headers[16:]
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

def find_sites_and_dates(observations):
    floatified = []
    sites_and_dates = []
    for row in observations:
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d").date()
        tick_count_in_site = float(row[3])
        if tick_count_in_site < 100:
            all_observations.append(row)
            floatify = [float(item) for item in row[3:]]
            floatified.append(floatify)
            sites_and_dates.append((row[1], date))
    return sites_and_dates

def show_per_site_model_performance_one_to_one(sites_and_dates, y_pred, ytrain):
    path_fig = r"\\ad.utwente.nl\home\garciamartii\Documents\PhD\Papers\Journals\02_IJGIS\images\print_v3\{0}"
    name_fig = "Figure_03_General_Performance_per_site.png"
    shortYearFmt = DateFormatter("%y")
    print(len(y_pred), len(ytrain), len(sites_and_dates))
    for i in range(len(sites_and_dates)):
        a_pred = y_pred[i]
        a_train = ytrain[i]
        newtuple = sites_and_dates[i] + (a_pred, a_train)
        sites_and_dates[i] = newtuple

    l = list(sorted(sites_and_dates, key=lambda x: x[0]))
    dic_sites = defaultdict(list)
    for tup in l:
        key = tup[0]
        dic_sites[key].append(tup[1:])

    for key in sorted(dic_sites.keys()):
        sorted_dates = list(sorted(dic_sites[key], key=lambda x:x[0]))
        dic_sites[key] = sorted_dates
    print("Total sites: ", len(dic_sites.keys()))

    i = 1
    dicscores = []
    plt.suptitle("Performance of RF predicting the AQT for each individual flagging site", fontsize=36)
    plt.subplots_adjust(hspace=.6, wspace=.6)
    for key in sorted(dic_sites.keys()):
        the_site = dic_sites[key]
        the_pred = np.array([item[1] for item in the_site])
        the_obs = np.array([item[2] for item in the_site])

        sort_dates, sort_target, sort_fit = zip(*sorted(zip(the_site, the_obs, the_pred)))
        score = np.round(r2_score(sort_target, sort_fit), decimals=2)
        dicscores.append([key, score])

        ax = plt.subplot(5, 6, i)
        plt.title("{0} (R2: {1})".format(key, score), size=16, fontweight="bold", y=1.08)
        plt.ylim(0,100)

        if i in [13]:
            ax.yaxis.labelpad = 20
            plt.ylabel("True values", size=30)
        if i in [27]:
            ax.xaxis.set_label_coords(1.1, -0.5)
            plt.xlabel("Predictions", size=30)

        plt.plot(the_pred, the_obs, 'o', label=u'Observations', color="darkblue", markeredgewidth=1.0, markeredgecolor='black')
        plt.plot(the_obs, the_obs, "r-", label="1:1", linewidth=2)

        ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
        ax.grid(b=True, which='major', color='#A8A8A8', linewidth=0.5)
        ax.grid(b=True, which='major', color='#A8A8A8', linewidth=0.5)

        i+=1

    # plt.savefig(path_fig.format(name_fig), dpi=300)
    plt.show()
    return dicscores

def goodness_of_fit(scores):
    plt.clf()
    sorted_scores = list(reversed(sorted(scores, key=itemgetter(1))))
    places = [item[0] for item in sorted_scores]
    places_cut = [item[0:3]+item[-2:] for item in places]
    r2 = [item[1] for item in sorted_scores]
    xlinspace = np.linspace(0, len(places)-1, len(places))
    plt.subplots_adjust(hspace=.3, wspace=.8)
    ax = plt.subplot(2, 2, 1)
    plt.title("Goodness of fit per flagging site", fontsize=30)
    plt.plot(xlinspace, r2)
    plt.grid(b=True, which='major', color='#A8A8A8', linewidth=0.5)
    plt.grid(b=True, which='major', color='#A8A8A8', linewidth=0.5)
    plt.xticks(xlinspace, places_cut, rotation=70, size="medium")
    plt.xlabel("Flagging sites", size=18)
    plt.ylabel("Coefficient of determination (R2)", size=18)
    ax.yaxis.labelpad = 20
    ax.xaxis.labelpad = 20
    plt.subplot(2, 2, 2)
    plt.subplot(2, 2, 3)
    plt.subplot(2, 2, 4)
    plt.show()


def bound_target(path):
    l = []
    k = 1
    idx = np.genfromtxt(path_in, delimiter=";", usecols=range(0,4), dtype=str, skip_header=1)
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

    for row in idx:
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
        value = float(row[3])
        low_whisker = dicwhiskers[date.month][0]
        upp_whisker = dicwhiskers[date.month][1]
        low_bound = low_whisker[0]
        upp_bound = upp_whisker[1]
        Y_value_bounded = np.divide((value - low_bound), (upp_bound - low_bound))
        l.append(Y_value_bounded)

    return [np.array(l), dicwhiskers, idx]


def bounded_descaling(ypred, dicwhiskers, idx, lim):
    i = 0
    l = []
    test_metadata = idx[lim:, :]
    for row in test_metadata:
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
        low_whisker = dicwhiskers[date.month][0]
        upp_whisker = dicwhiskers[date.month][1]
        low_bound = low_whisker[0]
        upp_bound = upp_whisker[1]
        Y_value_bounded = np.round(ypred[i] * (upp_bound - low_bound) + low_bound)
        l.append(Y_value_bounded)
        i += 1
    return np.array(l)

def bounded_descaling_NL(ypred, dicwhiskers, month):
    # print("Month: ", month)
    low_whisker = dicwhiskers[month][0]
    upp_whisker = dicwhiskers[month][1]
    # print("\t These whiskers: ", low_whisker, upp_whisker)
    low_bound = low_whisker[0] # the lowest of the lower, acts as 'minimum'
    upp_bound = upp_whisker[1] # the uppest of the upper, acts as 'maximum'
    # print("\t These bounds: ", low_bound, upp_bound)
    Y_bounded_desc = np.round(ypred * (upp_bound - low_bound) + low_bound)
    return Y_bounded_desc


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


################
# Main Program #
################

version = 6

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_savitzky_golay.csv"
path_testing_tables = r"E:/exchange/csv/IJGIS/testing_tables/v{0}/testing_tables_v{0}/2014/"
path_pred_rf = r"E:/exchange/csv/IJGIS/testing_tables/v8/predictions_v8/2014/{0}"

tmp = "{0}: Raw data and without zeros ({1} test samples)"

m, all_observations, headers_list, combination, descale, descale_target = load_stuff(path_in, experiment=2)
labels = prepare_labels(all_observations)
headers = headers_list[:]

target_limits = descale[0]
maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))

target_bound, whiskers, idx = bound_target(path_in)

# Read and prepare data
xtrain = m[:lim, 1:]
ytrain = target_bound[:lim]
xtest = m[lim:, 1:]
ytest = target_bound[lim:]

print("\nxtrain: {0} \t\t ytrain: {1} \t\t xtest: {2} \t\t ytest: {3}".format(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape))
print("\nRANDOM FOREST MODELLING")
print("-" * 30, "\n")
print("Training observation samples: ", xtrain.shape)
print("Training target samples: ", ytrain.shape)
print("Starting...")

rf = RandomForestRegressor(n_estimators=500, n_jobs=4, max_features="auto", oob_score=False, bootstrap=True,
                           criterion="mse", max_leaf_nodes=None, max_depth=None, min_samples_leaf=1,
                           min_samples_split=2, warm_start=False)

print("Created ensemble...")
rf.fit(xtrain, ytrain)
pred_rf = rf.predict(xtest)

for subdir, dirs, files in os.walk(path_testing_tables.format(version)):
    for file in files:
        print("\n\n")
        print("-" * 80)
        print("Testing file:  \t", file)
        print("-" * 80)
        name = tune_name(file)

        c, b, year, month, day = name[:-4].split("_")

        location_test_file = path_testing_tables.format(version)+ file
        xraw = np.loadtxt(location_test_file, skiprows=1, delimiter=";", usecols=range(3, 89))
        idx = np.loadtxt(location_test_file, skiprows=1, delimiter=";", usecols=range(0, 3))

        xtest = scale_NL(xraw)
        xtest[np.isnan(xtest)] = 0
        pred_rf = rf.predict(xtest).reshape(-1, 1)
        # pred_rf_descaled = descaling(pred_rf, target_limits).reshape(-1, 1)
        pred_rf_bounded_desc = bounded_descaling_NL(pred_rf, whiskers, int(float(month)))
        out_rf = np.hstack((idx, pred_rf_bounded_desc))
        np.savetxt(path_pred_rf.format(name), out_rf, delimiter=";")

# print("Descaling predictions and target")
# pred_rf_desc = bounded_descaling(pred_rf, whiskers, idx, lim)
# ytest_desc = bounded_descaling(ytest, whiskers, idx, lim)
#
# print("Metrics and indicators: ")
# rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_desc, pred_rf_desc)
# textstr = embed_metrics(rmse, rmses, rmseu, nrmse)
# stamp("Random Forest", rmse, rmses, rmseu, nrmse, mae, r2)
# plot_the_model("RF", pred_rf_desc, ytest_desc, labels[lim:], 1)
# show_FI(rf, combination, headers, 20)
#
# plt.show()
# plt.clf()
#
# sites_and_dates = find_sites_and_dates(all_observations[lim:])
# scores = show_per_site_model_performance_one_to_one(sites_and_dates, pred_rf_desc, ytest_desc)
# goodness_of_fit(scores)