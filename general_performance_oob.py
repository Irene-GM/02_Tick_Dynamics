import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils import *
from evaluate import evaluate
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from operator import itemgetter

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

def plot_model(name, ax, predictions, test):
    ax.set_title(tmp.format(name,len(ytest_desc)), size=14)
    ax.plot(predictions, test, 'o', label=u'Observations', color="#7F7E7A")
    # Add textbox summarizing the metrics used
    ax.annotate(textstr, xy=(1, 0), xycoords='axes fraction', fontsize=14,
        xytext=(-10, 10), textcoords='offset points', ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.2', fc='gray', alpha=0.3),)

    ax.set_xlim(0, 90)
    ax.plot(predictions, test, "o")
    ax.plot(predictions, predictions, "-")
    ax.grid()

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

################
# Main Program #
################

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_savitzky_golay.csv"
tmp = "{0}: predictions vs test ({1} samples)"

m, all_observations, headers_list, combination, descale, descale_target = load_stuff(path_in, experiment=0)
headers = headers_list[0:102] + ["yminusone"]#, "yminustwo"] # Like this because there is a weird symbol
target_limits = descale[0]
maxlim = m.shape[0]

xtrain_oob = m[:, 1:]
ytrain_oob = m[:, 0]

print("\nxtrain: {0} \t\t ytrain: {1}".format(xtrain_oob.shape, ytrain_oob.shape))
print("\nRANDOM FOREST MODELLING")
print("-" * 30, "\n")
print("Training observation samples: ", xtrain_oob.shape)
print("Training target samples: ", ytrain_oob.shape)
print("Starting...")

rf = RandomForestRegressor(n_estimators=500,
                           n_jobs=4,
                           max_features="auto",
                           oob_score=True,
                           bootstrap=True,
                           criterion="mse",
                           max_leaf_nodes=None, # This goes in pairs with max_depth
                           max_depth=None, # and None means maximum development of trees
                           min_samples_leaf=1,
                           min_samples_split=1,
                           warm_start=False)
print("Created ensemble...")
rf.fit(xtrain_oob, ytrain_oob)
pred_rf_oob = rf.oob_prediction_

print("\nGRADIENT BOOSTING REGRESSION MODELLING")
print("-" * 30, "\n")
print("Training observation samples: ", xtrain_oob.shape)
print("Training target samples: ", ytrain_oob.shape)
print("Starting...")
gbr = GradientBoostingRegressor(loss='huber',
                                alpha=0.95,
                                learning_rate=.05,
                                max_features=None,
                                max_leaf_nodes=None,
                                criterion="mse",
                                min_samples_split=2,
                                min_samples_leaf=3,
                                n_estimators=500,
                                presort=False,
                                subsample=0.2,
                                max_depth=5)
print("Training...\n")
gbr.fit(xtrain_oob, ytrain_oob)


# compute test set deviance
test_deviance = np.zeros((gbr.get_params()['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(gbr.staged_decision_function(xtrain_oob)):
    # clf.loss_ assumes that y_test[i] in {0, 1}
    test_deviance[i] = gbr.loss_(ytrain_oob, y_pred)

plt.plot((np.arange(test_deviance.shape[0]) + 1)[::5], test_deviance[::5], '-', linewidth=3, color="blue", label="GBR Deviance")
plt.grid()
plt.xlabel('Boosting Iterations')
plt.ylabel('Test Set Deviance')



print("OOB improvement: ", gbr.oob_improvement_)
print("Training Score: ", gbr.train_score_)

# plt.title("n_estimators=1000, subsample=0.2")
# plt.plot(np.linspace(0, 399, 400), gbr.oob_improvement_, "-", label="OOB improvement")
# plt.plot(np.linspace(0, 399, 400), gbr.train_score_, "-", label="Train score")
# plt.legend(loc='upper left')
# plt.show()
print("Predicting")
pred_gbr = gbr.predict(xtrain_oob)

print("Descaling predictions and target")
pred_rf_desc = np.round(descaling(pred_rf_oob, target_limits), decimals=0)
pred_gbr_desc = np.round(descaling(pred_gbr, target_limits), decimals=0)
ytest_desc = np.round(descaling(ytrain_oob, target_limits), decimals=0)

 # Order: rmse, rmses, rmseu, nrmse, mae, r2, the_mean, the_mean_pred

evaluate_gbr = show_model_evaluation(ytest_desc, pred_gbr_desc)
general_checks(ytrain_oob, ytest_desc, pred_rf_oob, pred_rf_desc, pred_gbr, pred_gbr_desc)

print("Metrics and indicators: ")
f, (ax1, ax2) = plt.subplots(2, 1, sharey=True)
rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_desc, pred_rf_desc)
textstr = embed_metrics(rmse, rmses, rmseu, nrmse)
stamp("Random Forest", rmse, rmses, rmseu, nrmse, mae, r2)
plot_model("RF", ax1, pred_rf_desc, ytest_desc)
show_FI(rf, combination, headers, 20)
# add_labels(ax1, all_observations, lim, pred_rf_desc, ytest_desc)

rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_desc, pred_gbr_desc)
textstr = embed_metrics(rmse, rmses, rmseu, nrmse)
stamp("Gradient Boosting", rmse, rmses, rmseu, nrmse, mae, r2)
plot_model("GBR", ax2, pred_gbr_desc, ytest_desc)
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






