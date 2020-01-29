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
    plt.subplot(3, 5, number)
    plt.title(tmp.format(name,len(ytest_desc)), size=14)
    plt.plot(predictions, test, 'o', label=u'Observations', color="#7F7E7A")
    # Add textbox summarizing the metrics used
    plt.annotate(textstr, xy=(1, 0), xycoords='axes fraction', fontsize=14,
        xytext=(-10, 10), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='gray', alpha=0.3),)

    # for label, x, y in zip(labels, predictions, test):
    #     # print(label, x, y)
    #     plt.annotate(label, xy = (x, y))

    plt.plot(predictions, test, "o")
    plt.plot(predictions, predictions, "-")
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


################
# Main Program #
################

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_savitzky_golay.csv"
tmp = "{0}: ({1} samples)"

names = ["Appelscha","Bilthoven", "Dronten", "Ede",
         "Eijsden", "Gieten", "HoogBaarlo", "KwadeHoek",
         "Montferland", "Nijverdal", "Schiermonnikoog",
         "Schouwen", "Twiske", "Vaals", "Veldhoven","Wassenaar"]

plotn = 1
plt.suptitle("GBR: Model performance when trained with TRANSECTS #1 and tested over TRANSECTS #2", size=14)
for name in names:
    print("Working with ", name)
    m1, m2, all_observations, headers_list, combination, descale1, descale2 = load_stuff_per_site(path_in, name, experiment=0)
    # headers = headers_list[0:102] # + ["doy"]# + ["s_tmin","i_tmin","s_tmax","i_tmax","s_prec","i_prec","s_ev","i_ev","s_rh","i_rh"]

    if m1 != None:
        target_limits1 = descale1[0]
        target_limits2 = descale2[0]
        print(m1.shape, m2.shape)
        labels = prepare_labels(all_observations)
        maxlim = m2.shape[0]
        lim = int(np.divide(maxlim * 70, 100))
        headers = headers_list[:]

        # Read and prepare data
        xtrain = m1[:, 1:]
        ytrain = m1[:, 0]
        xtest = m2[:, 1:]
        ytest = m2[:, 0]

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
                                        presort=True,
                                        subsample=1.0,
                                        max_depth=3)
        print("Training...\n")
        gbr.fit(xtrain, ytrain)

        print("Predicting")
        pred_gbr = gbr.predict(xtest)

        print("Descaling predictions and target")
        pred_rf_desc = np.round(descaling(pred_rf, target_limits2), decimals=0)
        pred_gbr_desc = np.round(descaling(pred_gbr, target_limits2), decimals=0)
        ytest_desc = np.round(descaling(ytest, target_limits2), decimals=0)

        evaluate_gbr = show_model_evaluation(ytest_desc, pred_gbr_desc)
        general_checks(ytest, ytest_desc, pred_rf, pred_rf_desc, pred_gbr, pred_gbr_desc)

        print("Metrics and indicators: ")
        rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_desc, pred_rf_desc)
        textstr = embed_metrics(rmse, rmses, rmseu, nrmse)
        # stamp("Random Forest", rmse, rmses, rmseu, nrmse, mae, r2)
        # plot_model(name, pred_rf_desc, ytest_desc, labels[lim:], plotn)
        # show_FI(rf, combination, headers, 20)
        # add_labels(ax1, all_observations, lim, pred_rf_desc, ytest_desc)


        # rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_desc, pred_gbr_desc)
        # textstr = embed_metrics(rmse, rmses, rmseu, nrmse)
        # stamp("Gradient Boosting", rmse, rmses, rmseu, nrmse, mae, r2)
        plot_model(name, pred_gbr_desc, ytest_desc, labels[lim:], plotn)
        # show_FI(gbr, combination, headers, 20)
        # # add_labels(ax2, all_observations, lim, pred_gbr_desc, ytest_desc)

        plotn += 1

plt.show()





