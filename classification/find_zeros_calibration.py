import csv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.model_selection import GridSearchCV

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



def report(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")



################
# Main program #
################


path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_raw_binarized.csv"

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


# use a full grid over all parameters
param_grid = {"max_depth": [3, 10, 20, None],
              "max_features": [3, 10, 20, 50, "auto", "sqrt", "log2", None],
              "min_samples_split": [1, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}


rf = RandomForestClassifier(n_estimators=300)
grid_search = GridSearchCV(rf, param_grid=param_grid)
start = time()
grid_search.fit(xtrain, ytrain)

print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)



# GridSearchCV took 32507.63 seconds for 1152 candidate parameter settings.
# Model with rank: 1
# Mean validation score: 0.803 (std: 0.021)
# Parameters: {'bootstrap': True, 'min_samples_split': 1, 'criterion': 'entropy', 'min_samples_leaf': 10, 'max_features': 50, 'max_depth': 10}
#
# Model with rank: 2
# Mean validation score: 0.802 (std: 0.020)
# Parameters: {'bootstrap': True, 'min_samples_split': 3, 'criterion': 'gini', 'min_samples_leaf': 10, 'max_features': 50, 'max_depth': 20}
#
# Model with rank: 2
# Mean validation score: 0.802 (std: 0.020)
# Parameters: {'bootstrap': True, 'min_samples_split': 10, 'criterion': 'entropy', 'min_samples_leaf': 10, 'max_features': 50, 'max_depth': 10}
#
# Model with rank: 4
# Mean validation score: 0.799 (std: 0.020)
# Parameters: {'bootstrap': True, 'min_samples_split': 1, 'criterion': 'entropy', 'min_samples_leaf': 10, 'max_features': 50, 'max_depth': 20}
#
# Model with rank: 5
# Mean validation score: 0.799 (std: 0.019)
# Parameters: {'bootstrap': True, 'min_samples_split': 1, 'criterion': 'entropy', 'min_samples_leaf': 10, 'max_features': 50, 'max_depth': None}
#
# Model with rank: 6
# Mean validation score: 0.798 (std: 0.020)
# Parameters: {'bootstrap': True, 'min_samples_split': 1, 'criterion': 'entropy', 'min_samples_leaf': 10, 'max_features': None, 'max_depth': 10}
#
# Model with rank: 6
# Mean validation score: 0.798 (std: 0.019)
# Parameters: {'bootstrap': True, 'min_samples_split': 3, 'criterion': 'entropy', 'min_samples_leaf': 10, 'max_features': None, 'max_depth': 10}
#
# Model with rank: 6
# Mean validation score: 0.798 (std: 0.023)
# Parameters: {'bootstrap': True, 'min_samples_split': 10, 'criterion': 'entropy', 'min_samples_leaf': 10, 'max_features': 20, 'max_depth': None}
#
# Model with rank: 6
# Mean validation score: 0.798 (std: 0.017)
# Parameters: {'bootstrap': True, 'min_samples_split': 1, 'criterion': 'entropy', 'min_samples_leaf': 10, 'max_features': None, 'max_depth': None}
#
# Model with rank: 10
# Mean validation score: 0.798 (std: 0.018)
# Parameters: {'bootstrap': True, 'min_samples_split': 1, 'criterion': 'gini', 'min_samples_leaf': 10, 'max_features': None, 'max_depth': None}
#
# Model with rank: 10
# Mean validation score: 0.798 (std: 0.023)
# Parameters: {'bootstrap': True, 'min_samples_split': 3, 'criterion': 'entropy', 'min_samples_leaf': 10, 'max_features': 50, 'max_depth': 20}




