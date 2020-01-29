import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
from evaluate import evaluate
import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, r2_score
import matplotlib.pyplot as plt
from collections import defaultdict

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


def count_zeros_ones(arr):
    zeros = 0
    for elem in arr.tolist():
        if elem[0]==0:
            zeros += 1
    ones = len(arr) - zeros
    return zeros, ones

def stamp_totals(zeros, ones):
    n = zeros + ones
    percent_z = np.round(np.divide(zeros * 100, n), decimals=1)
    percent_o = np.round(np.divide(ones * 100, n), decimals=1)
    print("Total Zeros: ", zeros, "\t{0}%".format(percent_z))
    print("Total Ones: ", ones, "\t{0}%".format(percent_o))
    print("Total Samples: ", n)

def stamp_metrics(ytrain, ytest):

    pack = [ytrain.reshape(-1, 1), ytest.reshape(-1, 1)]
    labels = ["Ytrain: ", "Ytest: "]
    total_zeros = 0
    total_ones = 0
    i = 0
    for item in pack:
        arr = item
        zeros, ones = count_zeros_ones(arr)
        total_zeros += zeros
        total_ones += ones
        print(labels[i])
        print("\t\t Zeros: ", zeros)
        print("\t\t Ones: ", ones)
        print("\t\t Shape: ", arr.shape)
        print()
        i+=1

    stamp_totals(total_zeros, total_ones)


def read_file(path_in):
    l = []
    idxs = []
    with open(path_in, "r", newline="") as r:
        reader = csv.reader(r, delimiter=";")
        next(reader)
        for row in reader:
            idxs.append(row[0])
            if float(row[3]) <= 100:
                floatify = [float(item) for item in row[3:]]
                l.append(floatify)

    m = np.array(l)

    return m, np.array(idxs)

def create_mask(idxs, decision_func):
    l =[]
    i = 0
    for row in decision_func:
        idx = idxs[i]
        class_zero = row[0]
        class_one = row[1]
        if class_zero >= class_one:
            l.append([int(idx), int(0)])
        else:
            l.append([int(idx), int(1)])
        i += 1
    return np.array(l)

def find_tokeep(mask, tomask):
    l = []
    dicmask = defaultdict(int)
    for row in mask:
        key = row[0]
        ismasked = row[1]
        dicmask[key] = ismasked
    for row in tomask:
        newkey = row[0]
        l.append(int(dicmask[newkey]))
    return np.array(l)


################
# Main program #
################


path_cla = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_raw_binarized.csv"
path_reg = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_savitzky_golay.csv"
path_out = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_mask.csv"

mcla, idxs_cla = read_file(path_cla)
mreg, idxs_reg = read_file(path_reg)

maxlim_cla = mcla.shape[0]
lim_cla = int(np.divide(maxlim_cla * 70, 100))

maxlim_reg = mreg.shape[0]
lim_reg = int(np.divide(maxlim_reg * 70, 100))

# Read and prepare data
xtrain_cla = mcla[:, 1:]
ytrain_cla = mcla[:, 0]

xtrain_reg = mreg[:lim_reg, 1:]
ytrain_reg = mreg[:lim_reg, 0]
xtest_reg = mreg[lim_reg:, 1:]
ytest_reg = mreg[lim_reg:, 0]

print("xtrain_cla: {0} \tytrain_cla: {1}".format(xtrain_cla.shape, ytrain_cla.shape))
print("xtrain_reg: {0} \tytrain_reg: {1} \txtest_reg: {2} \tytest_reg: {3}".format(xtrain_reg.shape, ytrain_reg.shape, xtest_reg.shape, ytest_reg.shape))

rfcla = RandomForestClassifier(n_estimators=500, oob_score=True)
rfcla.fit(xtrain_cla, ytrain_cla)
print("This is the OOB score: ", rfcla.oob_score_)

mask_cla = create_mask(idxs_cla, rfcla.oob_decision_function_)

print(mask_cla)

np.savetxt(path_out, mask_cla, delimiter=";", fmt="%d")

#
# rfreg = RandomForestRegressor(n_estimators=250,
#                            n_jobs=4,
#                            max_features="auto",
#                            oob_score=False,
#                            bootstrap=True,
#                            criterion="mse",
#                            max_leaf_nodes=None, # This goes in pairs with max_depth
#                            max_depth=None, # and None means maximum development of trees
#                            min_samples_leaf=1,
#                            min_samples_split=1,
#                            warm_start=False)
#
# rfreg.fit(xtrain_reg, ytrain_reg)
# pred_rf_reg = rfreg.predict(xtest_reg)
#
# print(idxs_reg[lim_reg:].shape, pred_rf_reg.shape)
#
# tomask_reg = np.vstack((idxs_reg[lim_reg:], pred_rf_reg)).T
# tokeep = find_tokeep(mask_cla, tomask_reg)
#
# print(tokeep.shape)
#
# product = tokeep * pred_rf_reg

# plt.hist(product, bins=20)
# plt.show()
#
# print(product.tolist()[0:30])
# print(ytest_reg.tolist()[0:30])
# print(tokeep[0:30])
#
# print(product.shape, ytest_reg.shape)
# print(product.tolist().count(0), ytest_reg.tolist().count(0))
#
# rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(ytest_reg, pred_rf_reg)
#
# print("Part 1")
# print(rmse)
# print(rmses)
# print(rmseu)
# print(nrmse)
#
#
# i = 0
# newlist1 = []
# newlist2 = []
# for i in range(len(product)):
#     if product[i]!=0:
#         newlist1.append(product[i])
#         newlist2.append(ytest_reg[i])
#         i+=1
#
# rmse, rmses, rmseu, nrmse, mae, r2, mean, mean_pred = show_model_evaluation(np.array(newlist2), np.array(newlist1))
#
# print()
# print("Part 2")
# print(rmse)
# print(rmses)
# print(rmseu)
# print(nrmse)
#
# plt.xlim(0, 100)
# plt.ylim(0, 100)
# plt.plot(np.array(newlist1), np.array(newlist2), "o")
# plt.plot(np.array(newlist1), np.array(newlist1), "-")
# plt.show()
#
