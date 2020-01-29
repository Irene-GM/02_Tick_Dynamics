import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import csv
from collections import defaultdict
import datetime

path_in = r"\\ad.utwente.nl\home\garciamartii\Documents\workspace\Special\IJGIS\data\new\newvegnoisy\FS_previous_year_metrics.csv"
path_ticks = r"\\ad.utwente.nl\home\garciamartii\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_raw.csv"
path_out = r"\\ad.utwente.nl\home\garciamartii\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_without_zeros_raw_prevyear.csv"

dic = defaultdict(list)

with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    for row in reader:
        key = (row[0], int(row[1]))
        mean = float(row[2])
        maximum = float(row[3])
        dic[key] = [mean, maximum]

with open(path_ticks, "r", newline="") as r:
    with open(path_out, "w", newline="") as w:
        reader = csv.reader(r, delimiter=";")
        headers = next(reader) + ["mean", "maximum", "std", "mode"]
        writer = csv.writer(w, delimiter=";")
        writer.writerow(headers)
        for row in reader:
            site = row[1]
            date = datetime.datetime.strptime(row[2], "%Y-%m-%d")

            if date.year == 2006:
                newkey = (site, date.year)
            else:
                newkey = (site, date.year-1)

            try:
                mean, maximum = dic[newkey]
            except ValueError:
                newkey = (site, date.year)
                mean, maximum = dic[newkey]

            newrow = row + [mean, maximum]
            writer.writerow(newrow)



