import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import csv
import datetime
from collections import defaultdict
import numpy as np
from scipy.stats.mstats import mode

path_in = r"\\ad.utwente.nl\home\garciamartii\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_raw.csv"
path_out = r"\\ad.utwente.nl\home\garciamartii\Documents\workspace\Special\IJGIS\data\new\newvegnoisy\FS_previous_year_metrics.csv"

dic = defaultdict(list)
with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    headers = next(reader)
    for row in reader:
        site = row[1]
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
        tickcount = float(row[3])
        key = (site, date.year)
        dic[key].append(tickcount)


with open(path_out, "w", newline="") as w:
    writer = csv.writer(w, delimiter=";")
    for key in sorted(dic.keys()):
        mean = np.round(np.mean(np.array(dic[key])), decimals=2)
        maximum = max(dic[key])
        std = np.round(np.std(np.array(dic[key])), decimals=2)
        themode = np.round(mode(dic[key])[0], decimals=2)
        row = [key[0], key[1], mean, maximum, std, themode]
        writer.writerow(row)


