import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from collections import defaultdict
from evaluate import evaluate
from sklearn.metrics import r2_score


def plot_model(name, predictions, test, labels, number):
    plt.subplot(2, 1, number)
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


def boxcox(arr, lmbda):
    lv = []
    for value in arr:
        if lmbda == 0:
            v = np.divide(np.power(value, lmbda), lmbda)
            lv.append(v)
        else:
            v = np.log(value)
            lv.append(v)

    bc = np.array(lv)
    bc[np.isnan(bc)] = 0
    bc[np.isinf(bc)] = 0
    return bc

def multiplicative_columns(xdays, vegetation):
    i = 0
    l = []
    lall = []
    for elem in xdays:
        for elem2 in vegetation:
            newvalue = elem * elem2
            l.append(newvalue)
        lall = lall + l
        l = []
    return lall


def show_model_evaluation(ytest, y_pred):
    evaluation = evaluate(ytest, y_pred)
    # rmse = np.sqrt(mean_squared_error(ytest, y_pred))
    rmse = evaluation[-4]
    rmses = evaluation[-3]
    rmseu = evaluation[-2]
    the_mean = np.mean(ytest)
    the_mean_pred = np.mean(y_pred)
    nrmse = np.divide(rmse, the_mean)
    # mae = mean_absolute_error(ytest, y_pred)
    mae = evaluation[-6]
    r2 = r2_score(ytest, y_pred)
    pack = [rmse, rmses, rmseu, nrmse, mae, r2, the_mean, the_mean_pred]
    return pack


def stamp(name, rmse, rmses, rmseu, nrmse, mae, r2):
    print("\n {0}".format(name))
    print("-"*30, "\n")
    print("RMSE (ytest and y_pred): ", rmse)
    print("\tRMSE systematic: ", rmses)
    print("\tRMSE UNsystematic: ", rmseu)
    print("\tNRMSE: ", nrmse)
    print("\nMAE (ytest and y_pred):", mae)
    print("R2 coefficient (ytest and y_pred: ", r2)


################
# Main program #
################


path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_savitzky_golay.csv"
path_flag = r'M:\Documents\workspace\Special\IJGIS\data\flagging_complete_34.csv'

dic = defaultdict(list)
with open(path_flag, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    for row in reader:
        place = row[1]
        transect = row[2]
        latitude = float(row[3])
        longitude = float(row[4])
        key = place + "_" + transect
        dic[key] = [latitude, longitude]


l = []
with open(path_in, "r", newline="") as r:
        headers = next(r)
        headers_list = headers.split(";")[3:]
        reader = csv.reader(r, delimiter=";")
        for row in reader:
            tickcount = [float(row[3])]
            if 0 < tickcount[0] < 100:
                place = row[1]
                mast = [float(item) for item in row[4:19]]
                xdays1 = [float(item) for item in row[19:26]]
                vegetation = [float(item) for item in row[97:103]]
                multiplicative = multiplicative_columns(xdays1, vegetation)
                landcover = [float(item) for item in row[103:]]
                newrow = tickcount + dic[place] + mast + xdays1 + vegetation + landcover # + multiplicative
                l.append(newrow)

m = np.array(l)
print(m.shape)

maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))

xtrain = m[:lim, 1:]
ytrain = m[:lim, 0]
xtest = m[lim:, 1:]
ytest = m[lim:,0]

ytrain[np.isnan(ytrain)] = 0
ytest[np.isnan(ytest)] = 0

print(np.isnan(ytrain).any(), np.isnan(ytest).any())
print(np.isinf(ytrain).any(), np.isinf(ytest).any())

print("\nxtrain: {0} \t\t ytrain: {1} \t\t xtest: {2} \t\t ytest: {3}".format(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape))
print("\nRANDOM FOREST MODELLING")
print("-" * 30, "\n")
print("Training observation samples: ", xtrain.shape)
print("Training target samples: ", ytrain.shape)
print("Starting...")

rf = RandomForestRegressor(n_estimators=500,
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

pred_rf_desc = pred_rf
ytest_desc = ytest

rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred  = show_model_evaluation(ytest_desc, pred_rf_desc)
stamp("Random Forest", rmse, rmses, rmseu, nrmse, mae, r2)

# rmse = np.sqrt(mean_squared_error(ytest_desc, pred_rf_desc))
# print("RMSE: ", rmse)
# print("NRMSE: ", np.divide(rmse, np.mean(pred_rf_desc)))

plt.plot(pred_rf_desc, ytest_desc, "o")
plt.plot(pred_rf_desc, pred_rf_desc, "-", color="red")
plt.show()
