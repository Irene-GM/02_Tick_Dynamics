import csv
import datetime
from collections import defaultdict
import numpy as np

path_fitting = r"M:\Documents\workspace\Special\IJGIS\data\new\newvegnoisy\savitzky_golay_fitting.csv"
path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\newvegnoisy\FS_nymphs_newVeg_noisy.csv"
path_out = r"M:\Documents\workspace\Special\IJGIS\data\new\newvegnoisy\FS_nymphs_newVeg_noisy_savitzky_golay.csv"

dic = defaultdict(list)
with open(path_in, "r", newline="") as r:
    # headers = next(r).split(";")
    # print(headers)
    reader = csv.reader(r, delimiter=";")
    next(reader)
    for row in reader:
        name = row[1]
        rowid = int(row[0])
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d").date()
        tick_counts = int(float(row[3]))
        floatified = [float(item) for item in row[4:]]
        newrow = [name, date, tick_counts] + floatified
        dic[rowid] = newrow


g = defaultdict(list)
with open(path_fitting, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    for row in reader:
        rowid = int(row[0])
        name = row[1]
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d").date()
        tick_counts_gauss = float(row[4])
        g[rowid] = [name, date, tick_counts_gauss]

with open(path_out, "w", newline="") as r:
    writer = csv.writer(r, delimiter=";")
    # writer.writerow(headers)
    for key in sorted(dic.keys()):
        fitrow = g[key]
        dicrow = dic[key]
        if fitrow[0:2] == dicrow[0:2]:
            r = key
            n = dicrow[0]
            d = dicrow[1]
            tick_gauss = np.round(fitrow[2], decimals=4)
            values = dicrow[3:]
            newrow = [r, n, d, tick_gauss] + values
            writer.writerow(newrow)

        else:
            print(fitrow[0:2], dicrow[0:2])
            print("Nope")
