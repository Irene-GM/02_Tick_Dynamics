import numpy as np
import datetime
from collections import defaultdict
import matplotlib.pyplot as plt

def scale_one(target):
    minimum = target.min(axis=0)
    maximum = target.max(axis=0)
    target_limits = [0, minimum, maximum]
    return [np.divide((target - minimum), (maximum - minimum)), target_limits]

def descale_one(target, target_limits):
    print("This is the target: ", target_limits)
    minimum = 0
    maximum = target_limits[2]
    X_descaled = np.round(target * (maximum - minimum) + minimum)
    return X_descaled

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_savitzky_golay.csv"

idx = np.genfromtxt(path_in, delimiter=";", usecols=range(0,4), dtype=str, skip_header=1)
dic = defaultdict(list)
for row in idx:
    date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
    value = float(row[3])
    dic[date.month].append(value)

data = [dic[i] for i in range(1, 13)]
ax = plt.boxplot(data)
# plt.show()

k = 1
dicwhiskers = defaultdict(tuple)
for i in range(2, 25, 2):
    lower_whisker = ax['whiskers'][i-2].get_ydata()
    upper_whisker = ax['whiskers'][i-1].get_ydata()
    dicwhiskers[k] = [lower_whisker, upper_whisker]
    k += 1

for key in dicwhiskers.keys():
    print(key, dicwhiskers[key])

l = []
for row in idx:
    value = float(row[3])
    date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
    upper_bound = dicwhiskers[date.month][1]
    if value > upper_bound[1]:
        # print("Correcting: ", value, upper_bound, date.month)
        corr = upper_bound[1]
    else:
        corr = value
    l.append(corr)

print(l[0:10])
scaled, target_limits = scale_one(np.array(l))
print(scaled[0:10])
descaled = descale_one(scaled, target_limits)
print(descaled[0:10])


