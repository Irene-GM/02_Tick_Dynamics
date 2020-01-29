import datetime
import csv
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import r2_score
import scipy.ndimage as ndimage


################
# Main program #
################

dic = defaultdict(list)

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\newveg\random_half_FS_nymphs_with_zeros_raw.csv"
path_out = r"M:\Documents\workspace\Special\IJGIS\data\new\newveg\index_random_half_FS_nymphs_with_zeros_gauss.csv"

with open(path_in, "r", newline="") as r:
    next(r)
    reader = csv.reader(r, delimiter=";")
    for row in reader:
        name = row[1]
        rowid = row[0]
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d").date()
        tick_counts = int(float(row[3]))
        dic[name].append([rowid, date, tick_counts])


# Its (7, 5) the maximum, with an avg R2 score of 0.61
# See full results below
# combi_sg = [(3, 2), (5, 2), (7, 2), (9, 2), (11, 2), (5, 3), (7, 3), (9, 3), (11, 3), (5, 4), (7, 4), (9, 4), (11, 4), (7, 5), (9, 5), (11, 5)]

i = 1
towrite = []

for key in sorted(dic.keys()):
    plt.subplot(3, 5, i)
    rowids = np.array([item[0] for item in dic[key]])
    dates = np.array([item[1] for item in dic[key]])
    target = np.array([item[2] for item in dic[key]])

    # fit_sg = savgol_filter(target, 7, 5)
    fit_gauss = ndimage.gaussian_filter(target, sigma=0.75, order=0)
    score = np.round(r2_score(target, fit_gauss), decimals=2)
    sort_dates, sort_target, sort_sg = zip(*sorted(zip(dates, target, fit_gauss)))
    plt.plot_date(sort_dates, sort_target, "-")
    plt.plot_date(sort_dates, sort_sg, "-", color="red")
    plt.grid()
    plt.ylim(0, 150)
    plt.title("{0} (R2: {1})".format(key, score))

    for j in range(len(fit_gauss)):
        r = int(rowids[j])
        k = key
        d = dates[j]
        t = target[j]
        f = np.round(fit_gauss[j], decimals=4)
        newrow = [r, k, d, t, f]
        towrite.append(newrow)

    i += 1

plt.show()

with open(path_out, "w", newline="") as w:
    writer = csv.writer(w, delimiter=";")
    for row in sorted(towrite):
        writer.writerow(row)






# Trying this combi:  (3, 2)
# 	 Average score:  1.0
# Trying this combi:  (5, 2)
# 	 Average score:  0.493333333333
# Trying this combi:  (7, 2)
# 	 Average score:  0.353333333333
# Trying this combi:  (9, 2)
# 	 Average score:  0.246666666667
# Trying this combi:  (11, 2)
# 	 Average score:  0.193333333333
# Trying this combi:  (5, 3)
# 	 Average score:  0.513333333333
# Trying this combi:  (7, 3)
# 	 Average score:  0.373333333333
# Trying this combi:  (9, 3)
# 	 Average score:  0.26
# Trying this combi:  (11, 3)
# 	 Average score:  0.206666666667
# Trying this combi:  (5, 4)
# 	 Average score:  1.0
# Trying this combi:  (7, 4)
# 	 Average score:  0.593333333333
# Trying this combi:  (9, 4)
# 	 Average score:  0.46
# Trying this combi:  (11, 4)
# 	 Average score:  0.353333333333
# Trying this combi:  (7, 5)
# 	 Average score:  0.606666666667
# Trying this combi:  (9, 5)
# 	 Average score:  0.473333333333
# Trying this combi:  (11, 5)
# 	 Average score:  0.373333333333