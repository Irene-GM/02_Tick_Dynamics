import csv
import matplotlib.pyplot as plt
import numpy as np


path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_raw.csv"

l = []

with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    next(reader)
    for row in reader:
        floatify = [float(item) for item in row[3:]]
        l.append(floatify)

    m = np.array(l)

plt.hist(m[:,0], bins=100)
plt.xlim(0, 150)
plt.show()
