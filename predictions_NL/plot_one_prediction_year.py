import os
import datetime
import csv
import matplotlib.pyplot as plt
import numpy as np

def generate_dates(year):
    basedate = datetime.datetime(year, 1, 1)
    for x in range(0, 365):
        increment = basedate + datetime.timedelta(days=x)
        yield(increment)

def format_ints(m, d):
    if m<10:
        mo = str(m).zfill(2)
    else:
        mo = str(m)
    if d<10:
        da = str(d).zfill(2)
    else:
        da = str(d)
    return mo, da


################
# Main program #
################

dicpixels = {'Schiermonnikoog_1': [206, 28], 'Twiske_1': [121, 144],
             'Montferland_1': [198, 206], 'Eijsden_1': [180, 312],
             'Gieten_2': [247, 80], 'Dronten_2': [175, 135], 'Wassenaar_1': [85, 177],
             'Appelscha_2': [218, 90],
             'Bilthoven_1': [144, 176], 'Ede_2': [176, 191], 'Gieten_1': [247, 80],
             'Nijverdal_2': [225, 156],
             'Dronten_1': [175, 135],
             'Wassenaar_2': [85, 177], 'De Kwade Hoek_1': [60, 212],
             'Appelscha_1': [218, 90], 'De Kwade Hoek_2': [60, 212],
             'Twiske_2': [121, 144], 'Hoog Baarlo_2': [188, 182],
             'Montferland_2': [198, 206],
             'Eijsden_2': [180, 316],
             'Ede_1': [176, 191], 'Veldhoven_1': [151, 258], 'Vaals_2': [185, 318],
             'Vaals_1': [185, 318],
             'Nijverdal_1': [225, 156], 'Veldhoven_2': [151, 258],
             'Schiermonnikoog_2': [206, 28], 'Bilthoven_2': [144, 176]}

path_in = r"/home/irene/PycharmProjects/NL_predictors/data/versions/v8/predictions_v8/2014/{0}"
basename = "NL_Prediction_{0}_{1}_{2}.csv"
year = 2014
l = []
for date in generate_dates(year):
    lyr = np.empty((350, 300))
    m, d = format_ints(date.month, date.day)
    name = basename.format(date.year, m, d)
    print("Reading ", name)
    with open(path_in.format(name), "r", newline="") as r:
        reader = csv.reader(r, delimiter=";")
        for row in reader:
            i = int(float(row[1]))
            j = int(float(row[2]))
            v = int(float(row[3]))
            lyr[i,j] = v
        l.append(lyr)

stack = np.dstack(l)


i = 1
plt.suptitle("Predicted AQT per flagging site during the year 2014")
plt.subplots_adjust(hspace=0.5, wspace=0.5)
for key in sorted(dicpixels.keys()):
    xpix = dicpixels[key][1]
    ypix = dicpixels[key][0]
    l = stack[xpix, ypix, :]
    plt.subplot(5, 6, i)
    # l30 = [l[i] for i in range(0, 365) if i%30==0]
    plt.plot(l, "-")
    plt.ylim(0, 80)
    plt.title(key)
    plt.grid()
    i += 1

plt.show()



