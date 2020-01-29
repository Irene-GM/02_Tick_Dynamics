import csv
import datetime
from collections import defaultdict
from operator import itemgetter

def find_previous(date, l):
    i = 0
    for tup in l:
        current = tup[1]
        # print("Current: ", current, date)
        if current >= date:
            return l[i-1]
        else:
            i+=1
    return [l[i]]

def find_previous_old(date, l):
    i = 0
    for tup in l:
        current = tup[1]
        if current >= date:
            yday = i-1
            beforeyday = i-2

            if yday>0 and beforeyday>0:
                return [l[yday], l[beforeyday]]

            elif yday>0 and beforeyday==0:
                return [l[yday], l[yday]]

            elif yday==0:
                return [l[i], l[i]]
        else:
            i+=1
    return [l[i], l[i]]


################
# Main program #
################

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\SG52\random_FS_All_Preds_nymphs_without_zeros_SG52.csv"
path_out = r"M:\Documents\workspace\Special\IJGIS\data\new\SG52\random_FS_All_Preds_nymphs_without_zeros_SG52_yminusone.csv"

dic = defaultdict(list)
complete_dic = []
with open(path_in, newline="") as r:
    reader = csv.reader(r, delimiter=";")
    next(reader)
    for row in reader:
        id = int(row[0])
        key = row[1]
        date = datetime.datetime.strptime(row[2], '%Y-%m-%d')
        today = float(row[3])
        tup = (id, date, today)
        dic[key].append(tup)
        complete_dic.append(row)

sorted_dic = defaultdict(list)
for key in dic.keys():
    l = dic[key]
    l.sort(key=itemgetter(1))
    sorted_dic[key] = l

append_dic = defaultdict(tuple)
for key in sorted_dic.keys():
    for tup in sorted_dic[key]:
        previous = find_previous(tup[1], sorted_dic[key])
        # append_dic[tup[0]] = [previous[0][2], previous[1][2]]
        append_dic[tup[0]] = [previous[2]]
        print(tup, "\t", previous)

with open(path_in, "r", newline="") as r:
    with open(path_out, "w", newline="") as w:
        writer = csv.writer(w, delimiter=";")
        reader = csv.reader(r, delimiter=";")
        next(reader)
        for row in reader:
            id = int(row[0])
            today = float(row[3])
            newrow = row + append_dic[id]
            writer.writerow(newrow)





