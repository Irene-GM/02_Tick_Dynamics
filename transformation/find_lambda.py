import csv
import numpy as np
import matplotlib.pyplot as plt

def boxcox(arr, lmbda):
    lv = []
    for value in arr:
        if lmbda == 0:
            v = np.divide(np.power(value, lmbda), lmbda)
            lv.append(v)
        else:
            v = np.log(value)
            lv.append(v)
    return np.array(lv)



################
# Main Program #
################
l = []
path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_savitzky_golay.csv"
with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    next(reader)
    for row in reader:
        v = float(row[3])
        if v>0:
            l.append(v)

plt.subplot(3, 3, 1)
plt.title("Raw data")
plt.hist(np.array(l), bins=25)

plt.subplot(3, 3, 2)
plt.title("Log + 1")
plusone = np.array(l) + 1
plt.hist(np.log(plusone), bins=25)

plt.subplot(3, 3, 3)
plt.title("Inverse square")
newtarget = np.divide(1, np.power(np.array(l), 2))
plt.hist(newtarget, bins=25)

plt.subplot(3, 3, 4)
plt.title("Reciprocal")
newtarget = np.divide(1, np.array(l))
plt.hist(newtarget, bins=25)

plt.subplot(3, 3, 5)
plt.title("Inverse square root")
newtarget = np.divide(1, np.sqrt(np.array(l)))
plt.hist(newtarget, bins=25)

plt.subplot(3, 3, 6)
plt.title("Square root")
newtarget = np.sqrt(np.array(l))
plt.hist(newtarget, bins=25)

plt.subplot(3, 3, 7)
plt.title("Square")
newtarget = np.power(np.array(l), 2)
plt.hist(newtarget, bins=25)

plt.subplot(3, 3, 8)
plt.title("Cubic")
newtarget = np.power(np.array(l), 2)
plt.hist(newtarget, bins=25)

plt.show()

for lmbda in np.arange(-2.5, 2.55, 0.05):
    pass
