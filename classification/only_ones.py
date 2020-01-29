import csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

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


################
# Main program #
################


path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_raw.csv"

l = []

with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    next(reader)
    for row in reader:
        tickcount = float(row[3])
        if tickcount>0:
            floatify = [float(item) for item in row[3:]]
            l.append(floatify)

m = np.array(l)
maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))

# Read and prepare data
xtrain = m[:lim, 1:]
ytrain = m[:lim, 0]
xtest = m[lim:, 1:]
ytest = m[lim:, 0]

print("xtrain: {0} \tytrain: {1} \txtest: {2} \tytest: {3}".format(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape))

rf = RandomForestRegressor(n_estimators=300,
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
rf.fit(xtrain, ytrain)
pred_rf = rf.predict(xtest)

print("R2: ", r2_score(ytest, pred_rf))

xlinspace = np.linspace(0, len(pred_rf)-1, len(pred_rf))
plt.plot(pred_rf, ytest, "o")
plt.plot(pred_rf, pred_rf, "-", color="black")
plt.show()




