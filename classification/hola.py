import numpy as np
from sklearn.ensemble import RandomForestClassifier
import csv
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from jenks import jenks
import matplotlib.pyplot as plt

def build_dic(mask):
    dic = defaultdict(int)
    for row in mask:
        key = int(row[0])
        val = int(row[1])
        dic[key] = val
    return dic

def goodness_of_variance_fit(array, classes):
    # get the break points
    classes = jenks(array, classes)
    # do the actual classification
    classified = np.array([classify(i, classes) for i in array])
    # max value of zones
    maxz = max(classified)
    # nested list of zone indices
    zone_indices = [[idx for idx, val in enumerate(classified) if zone + 1 == val] for zone in range(maxz)]
    # sum of squared deviations from array mean
    sdam = np.sum((array - array.mean()) ** 2)
    # sorted polygon stats
    array_sort = [np.array([array[index] for index in zone]) for zone in zone_indices]
    # sum of squared deviations of class means
    # subtraction = classified - classified.mean()
    sdcm = sum([np.sum((cla - cla.mean()) ** 2) for cla in array_sort])
    # goodness of variance fit
    gvf = (sdam - sdcm) / sdam
    return gvf

def classify(value, breaks):
    for i in range(1, len(breaks)):
        if value < breaks[i]:
            return i
    return len(breaks) - 1

def build_jenks(target):
    gvf = 0
    nclasses = 2
    while gvf < 0.95 and nclasses < 10:
        gvf = goodness_of_variance_fit(target, nclasses)
        print( "\tGVF for {0} classes: {1}".format(nclasses, gvf))
        nclasses += 1

    breaks = jenks(target, nclasses-1)
    print("Breaks: ", breaks)
    classified = np.array([classify(i, breaks) for i in target]).reshape(-1, 1)
    print(classified.shape)
    return classified

def process_decision_function(df):
    pred = []
    for row in df:
        maxvalue = max(row)
        pos = row.tolist().index(maxvalue)
        pred.append(pos)
    return np.array(pred)


################
# Main program #
################

path_mask = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_mask.csv"
path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_savitzky_golay.csv"
mask = np.loadtxt(path_mask, delimiter=";")

counter = 0

dic = build_dic(mask)

tokeep = []
with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    next(reader)
    for row in reader:
        key = float(row[0])
        tickcount = float(row[3])
        if tickcount < 100:
            if dic[key] == 1:
                floatify = [float(item) for item in row[3:]]
                tokeep.append(floatify)
            else:
                counter += 1

m = np.array(tokeep)
mcla = build_jenks(m[:,0])

plt.hist(mcla, bins=5)
plt.show()

print("Num. of zero samples: ", counter)
maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))

# Read and prepare data
xtrain = m[:lim, 1:]
ytrain = mcla[:lim, 0]
xtest = m[lim:, 1:]
ytest = mcla[lim:, 0]

print("The ytrain: ", ytest[0:10].tolist())
print("The ytest: ", ytest[0:10].tolist())

rf = RandomForestClassifier(n_estimators=300, oob_score=True)
rf.fit(xtrain, ytrain)
pred_rf = rf.predict(xtest)
# pred_rf = process_decision_function(rf.oob_decision_function_)
print(np.multiply(np.round(rf.oob_decision_function_[0:10, :], decimals=1), 100))
print(rf.oob_score_)

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


