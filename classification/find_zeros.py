import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix


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


path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_raw_binarized_ones.csv"

l = []

with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    next(reader)
    for row in reader:
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

print("ytrain 0 to 20")
print(ytrain[0:20])
print("ytest 0 to 20")
print(ytest[0:20])
print("xtrain[0]", xtrain[0,:])
print("xtest[0]", xtest[0,:])

stamp_metrics(ytrain, ytest)

rf = RandomForestClassifier(n_estimators=300,
                            max_features=50,
                            max_depth=10,
                            min_samples_leaf=10,
                            criterion="entropy",
                            min_samples_split=1,
                            bootstrap=True)


print("The ytrain: ", ytrain[0:30].tolist())
print("The ytest: ", ytest[0:30].tolist())

rf.fit(xtrain, ytrain)

pred_rf = rf.predict(xtest)

print("\nMetrics")
print("-"*40)

rf_sco = round(rf.score(xtest, ytest),2)
f1_sco = round(f1_score(ytest, pred_rf, average="macro"), 2)
prec_sco = round(precision_score(ytest, pred_rf, average="macro"), 2)
recall_sco = round(recall_score(ytest, pred_rf, average="macro"), 2)

print("RF Score: \t", rf_sco)
print("F1 Score: \t", f1_sco)
print("Precision: \t", prec_sco)
print("Recall: \t", recall_sco)
print()
print("Confusion Matrix")
print(confusion_matrix(ytest, pred_rf))
