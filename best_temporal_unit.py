import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils import *
from evaluate import evaluate
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from operator import itemgetter

def show_error_plots(nrmse_list, rmse_list, rmses_list, rmseu_list, r2_list, mae_list, labels):
    s = 16
    plt.clf()
    plt.close()
    xlinspace = np.linspace(0, 10, 11)

    plt.suptitle("GBR performance metrics at multiple temporal aggregations", size=22)

    plt.subplots_adjust(hspace=.5)
    plt.subplot(1, 2, 1)
    plt.plot(xlinspace, nrmse_list, "-", label="NRMSE", linewidth=2, color="black")
    plt.title("Evolution of NRMSE", size=s)
    plt.xticks(xlinspace, labels, size='small', rotation=70)
    plt.xlabel('Temporal aggreagation', size=s)
    plt.ylabel('Error value', size=s)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='-')

    plt.subplot(1, 2, 2)
    plt.plot(xlinspace, rmse_list, "-", color="black", label="RMSE", linewidth=2)
    plt.plot(xlinspace, rmses_list, "-", color="darkgray", label="RMSEs", linewidth=2)
    plt.plot(xlinspace, rmseu_list, "-", color="gray", label="RMSEu", linewidth=2)
    plt.title("Evolution of RMSE", size=s)
    plt.xlabel('Temporal aggreagation', size=s)
    plt.ylabel('Error value', size=s)
    plt.xticks(xlinspace, labels, size='small', rotation=70)
    plt.grid(b=True, which='major', color='lightgray', linestyle='-')
    plt.grid(b=True, which='minor', color='lightgray', linestyle='-')
    plt.legend(loc="lower right")

def show_feature_importances(rf, combination, headers, lim):
    cols = [headers[item] for item in combination[1:]] # Skip label tick count
    i = 1
    print("\nFeature Importances")
    print("-"*40)
    for item in list(reversed(sorted(zip(cols, rf.feature_importances_), key=lambda x: x[1])))[:lim]:
        print(i, ")\t", item[0], "\t\t", np.round(item[1] * 100, decimals=2), "%")
        i += 1

################
# Main program #
################

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_savitzky_golay.csv"

m, all_observations, headers_list, combinations, descale, descale_target = load_stuff(path_in, experiment=1)

target_limits = descale[0]
maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))

nrmse_list, rmse_list, rmses_list, rmseu_list, r2_list, mae_list = ([] for i in range(6))

labels = ["Xdays-1", "Xdays-2", "Xdays-3", "Xdays-4", "Xdays-5", "Xdays-6", "Xdays-7", "Xdays-14", "Xdays-30", "Xdays-90", "Xdays-365"]

lrf = []
lgbr = []

for combination in combinations:
    msel = select_columns(m, combination)
    xtrain = msel[:lim, 1:]
    ytrain = msel[:lim, 0]
    xtest = msel[lim:, 1:]
    ytest = msel[lim:, 0]

    print("Training...\n")


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
                                presort=False,
                                max_depth=5)

    gbr.fit(xtrain, ytrain)

    # I AM NOT CRAZY, GBR IS A STOCHASTIC GBR WITH SUBSAMPLING (BAGGING)
    # THUS I GUESS THIS IS THE ONLY WAY OF HAVING ACCESS TO THE PREDICTIONS
    # SINCE GBR HAS NO SUCH THING AS GBR.OOB_PREDICTIONS_ AS RANDOM FOREST

    pred_gbr = gbr.predict(xtest)

    pred_rf_desc = np.round(descaling(pred_rf, target_limits), decimals=0)
    pred_gbr_desc = np.round(descaling(pred_gbr, target_limits), decimals=0)
    ytest_desc = np.round(descaling(ytest, target_limits), decimals=0)

    # Order: rmse, rmses, rmseu, nrmse, mae, r2, the_mean, the_mean_pred

    # evaluate_gbr = show_model_evaluation(ytest_desc, pred_gbr_desc)
    # rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_desc, pred_gbr_desc)
    # nrmse_list.append(nrmse)
    # rmse_list.append(rmse)
    # rmses_list.append(rmses)
    # rmseu_list.append(rmseu)
    # r2_list.append(r2)
    # mae_list.append(mae)
    #
    # show_feature_importances(gbr, combination, headers_list, 5)

    evaluate_gbr = show_model_evaluation(ytest_desc, pred_rf_desc)
    rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_desc, pred_rf_desc)
    nrmse_list.append(nrmse)
    rmse_list.append(rmse)
    rmses_list.append(rmses)
    rmseu_list.append(rmseu)
    r2_list.append(r2)
    mae_list.append(mae)

    show_feature_importances(rf, combination, headers_list, 5)


show_error_plots(nrmse_list, rmse_list, rmses_list, rmseu_list, r2_list, mae_list, labels)
plt.show()
