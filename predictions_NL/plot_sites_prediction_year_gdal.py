import gdal
import datetime
import numpy as np
import matplotlib.pyplot as plt

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


gendates = generate_dates(2014)


path_in = r"/home/irene/PycharmProjects/NL_predictors/data/versions/v8/maps_v8/2014/{0}"
basename = "NL_Map_RF_NL_Prediction_{0}_{1}_{2}.tif"

l = []
for date in gendates:
    m, d = format_ints(date.month, date.day)
    name = basename.format(date.year, m, d)
    path = path_in.format(name)
    print(path)
    tif = gdal.Open(path)
    data = tif.GetRasterBand(1).ReadAsArray(225, 156, 1, 1)[0][0]
    l.append(data)

xlinspace = np.linspace(0, len(l)-1, len(l))
plt.plot(xlinspace, np.array(l), "-")
plt.show()


