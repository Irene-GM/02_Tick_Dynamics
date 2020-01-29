# -*- coding: utf-8 -*-

# This table uses data aggregated to a monthly
# resolution. Perhaps in the future it makes more
# sense to use a finer experiment_time_features resolution, but so
# far, this one is used to match the temporality
# of flagging datasets

# The original program was written in Python 2.
# This is an adapted version

# @author: GarciaMartiI


from osgeo import ogr, osr
import gdal
import csv
import datetime
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
import itertools
import matplotlib.pyplot as plt

years = [2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014]
variables = ["tmin", "tmax", "prec", "ev", "rh"]

# path_sites = r'D:\GeoData\Flagging\flagging_sites_DMS_only_complete_ts\flagging_sites_DMS_only_complete_ts.shp'
# path_flag = r'M:\Documents\workspace\Special\IJGIS\data\flagging_complete.csv'
# path_lc =  r"\\ad.utwente.nl\home\GarciaMartiI\Documents\workspace\Special\IJGIS\data\tables\flagging_sites_with_LC.csv"
# path_lc_big =  r"\\ad.utwente.nl\home\GarciaMartiI\Documents\workspace\Special\IJGIS\data\tables\flagging_sites_with_LC_500m_1km.csv"
# path_frag = r"\\ad.utwente.nl\home\GarciaMartiI\Documents\workspace\Special\IJGIS\data\tables\Flagging_Sites_Fragmentation_Indices_annotated.csv"
# path_deer = r"M:\Documents\workspace\DataProcessing\Environment\Deer\data\Flagging_Sites_Deer_clean.csv"

basepath = r"/datasets/KNMI/Xdays_v2/{0}/{1}_{2}_{3}.tif"
path_veg = r'/home/irene/PycharmProjects/NL_predictors/data/NL_signal_decomposition.csv'
path_lc = r"/home/irene/PycharmProjects/NL_predictors/data/Grid_NL_1km_LandCover.csv"
path_out = r'/home/irene/PycharmProjects/NL_predictors/data/versions/v6/testing_tables_v6/2010/NL_All_Predictors_LC_{0}_{1}_{2}.csv'


# basepath = r"F:\RSData\KNMI\Xdays_v2\{0}\{1}_{2}_{3}.tif"
# path_veg = r'M:\Documents\workspace\Special\IJGIS\data\NL\NL_signal_decomposition.csv'
# path_lc = r"M:\Documents\workspace\Special\IJGIS\data\NL\Grid_NL_1km_LandCover.csv"
# path_out = r'D:\GeoData\workspaceimg\Special\Paper2\testing_tables_v4\NL_All_Predictors_LC_{0}_{1}_{2}.csv'

cols_tickcount = ["TickCount"]
cols_weather = ["tmin-1", "tmax-1", "prec-1", "ev-1", "rh-1", "sd-1", "vp-1",
                "tmin-2", "tmax-2", "prec-2", "ev-2", "rh-2", "sd-2", "vp-2",
                "tmin-3", "tmax-3", "prec-3", "ev-3", "rh-3", "sd-3", "vp-3",
                "tmin-4", "tmax-4", "prec-4", "ev-4", "rh-4", "sd-4", "vp-4",
                "tmin-5", "tmax-5", "prec-5", "ev-5", "rh-5", "sd-5", "vp-5",
                "tmin-6", "tmax-6", "prec-6", "ev-6", "rh-6", "sd-6", "vp-6",
                "tmin-7", "tmax-7", "prec-7", "ev-7", "rh-7", "sd-7", "vp-7",
                "tmin-14", "tmax-14", "prec-14", "ev-14", "rh-14", "sd-14", "vp-14",
                "tmin-30", "tmax-30", "prec-30", "ev-30", "rh-30", "sd-30", "vp-30",
                "tmin-90", "tmax-90", "prec-90", "ev-90", "rh-90", "sd-90", "vp-90",
                "tmin-365", "tmax-365", "prec-365", "ev-365", "rh-365", "sd-365", "vp-365"]

# cols_habitat = ["Litter", "Moss", "Herb", "Brush", "Tree", "BioLC"]
# cols_mast = ["Eik-0", "Beuk-0", "Aeik-0", "Eik-1", "Beuk-1", "Aeik-1", "Eik-2", "Beuk-2", "Aeik-2"]
cols_date = ["Date"]
cols_veg = ["min_ndvi", "range_ndvi", "min_evi", "range_evi", "min_ndwi", "range_ndwi"]
cols_lc = ["LandCover", "LandCover_500m", "LandCover_1km"]
# cols_deer = ["Deer"]
# cols_frag = []

def saturation_deficit(rh, t):
    up = 0.0621 * t
    p1 = (1 - np.divide(rh, 100))
    p2 = 4.9463 * np.exp(up)
    r = p1 * p2
    return np.round(r, decimals=2)

def vapour_pressure_deficit(rh, t):
    up = np.divide(17.27 * t, 273.3 + t)
    p1 = (1 - np.divide(rh, 100))
    p2 = 0.611 * np.exp(up)
    r = p1 * p2
    return np.round(r, decimals=2)

def calculate_sd_vpd(l):
    lsd = []
    for i in range(11):
        ti = 5*i
        rhi = (5*i) + 4
        sd = saturation_deficit(l[rhi], l[ti])
        vpd = vapour_pressure_deficit(l[rhi], l[ti])
        lsd.append([sd, vpd])
    return lsd


# def saturation_deficit(rh, t):
#     t_div = np.divide(t, 100)
#     rh_div = np.divide(rh, 100)
#     up = 0.0621 * t_div
#     p1 = (1 - np.divide(rh_div, 100))
#     p2 = 4.9463 * np.exp(up)
#     r = p1 * p2
#     return np.round(r, decimals=2)
#
# def vapour_pressure_deficit(rh, t):
#     rh_div = np.divide(rh, 100)
#     t_div = np.divide(t, 100)
#     up = np.divide(17.27 * t_div, 273.3 + t_div)
#     p1 = (1 - np.divide(rh_div, 100))
#     p2 = 0.611 * np.exp(up)
#     r = p1 * p2
#     return np.round(r, decimals=2)
#
# def calculate_sd_vpd(l):
#     lsd = []
#     for i in range(11):
#         ti = 5*i
#         rhi = (5*i) + 4
#         if l[ti] != -99:
#             sd = np.round(saturation_deficit(l[rhi], l[ti]), decimals=2)
#             vpd = np.round(vapour_pressure_deficit(l[rhi], l[ti]), decimals=2)
#             lsd.append([sd, vpd])
#         else:
#             lsd.append([0, 0])
#
#     return lsd

def insert_in_order(l, lsd):
    chunked = []
    for i in range(11):
        ini = 5*i
        end = (5*i) + 5

        tmin = round(l[ini], 2)
        tmax = round(l[ini+1], 2)
        prec = round(l[ini+2], 2)
        ev = round(l[end-2], 2)
        rh = round(l[end-1], 2)

        # if rh != -99:
        #     # The following line divides the RH to be between 0-100
        #     rh = np.round(np.divide(rh, 100), decimals=2)
        #
        # if tmin != -99:
        #     # The following line divides tmin, tmax and ev
        #     tmin = np.round(np.divide(tmin, 100), decimals=2)
        #     tmax = np.round(np.divide(tmax, 100), decimals=2)
        # if ev != -99:
        #     ev = np.round(np.divide(ev, 100), decimals=2)
        #
        # if prec != -99:
        #     prec = np.round(np.divide(prec, 100), decimals=2)

        chunk = [tmin, tmax, prec, ev, rh] + lsd[i]
        chunked = chunked + chunk

    return chunked


def generateTifPathsWeather(date):
    timedeltas = ["01", "02", "03", "04", "05", "06", "07", "14", "30", "90", "365", ]
    year = date.year
    for i in range(len(timedeltas)):
        for variable in range(len(variables)):
            if year > 2005:
                what = variables[variable]
                timedelta_str = timedeltas[i]
                timedelta_folder = int(timedeltas[i])
                # yield(basepath.format(timedelta_folder, year, what, timedelta_str))
                yield(basepath.format(what, year, what, timedelta_str))

def extractSingleData(tif, x, y, numband):
    band = tif.GetRasterBand(numband)
    data = band.ReadAsArray(x, y, 1, 1)
    if data != None:
        if data[0][0] == -32768:
            return -99
        else:
            return data[0][0]
    else:
        return -99

def loadVegetation():
    dic = defaultdict(list)
    with open(path_veg, "r", newline="") as r:
        reader = csv.reader(r, delimiter=";")
        for row in reader:
            rowid = row[0]
            x = int(row[1])
            y = int(row[2])
            key = (x, y)
            # Instead of using minimum and maximum vegetation indices
            # what we do is using the minimum and the range, with the hope
            # it is more informative.
            min_ndvi = float(row[3])
            max_ndvi = float(row[4])
            range_ndvi = max_ndvi - min_ndvi

            min_evi = float(row[5])
            max_evi = float(row[6])
            range_evi = max_evi - min_evi

            min_ndwi = float(row[7])
            max_ndwi = float(row[8])
            range_ndwi = max_ndwi - min_ndwi
            newrow = [min_ndvi, range_ndvi, min_evi, range_evi, min_ndwi, range_ndwi]
            dic[key] = newrow
    return dic


def loadLandCover():
    dic = defaultdict(list)
    with open(path_lc, "r", newline="") as r:
        original_labels = []
        reader = csv.reader(r, delimiter=";")
        for row in reader:
            rowid = row[0]
            x = int(row[1])
            y = int(row[2])
            lc25 = int(row[3])
            lc500 = int(row[4])
            lc1000 = int(row[5])
            key = (x, y)
            dic[key] = [lc25, lc500, lc1000]

    return dic

def extract(x, y, date, dicveg, diclc):
    dic = defaultdict(list)
    doy = date.timetuple().tm_yday
    lweather, lveg, llc = ([] for i in range(3))

    for path in generateTifPathsWeather(date):
        tif = gdal.Open(path)
        data_in_point = extractSingleData(tif, y, x, doy)
#         print("File: ", path, "\t Data:", data_in_point, "\t DOY: ", doy)
        lweather.append(np.round(data_in_point, decimals=2))

    check_for_nans = np.isnan(np.array(lweather)).any()

    if check_for_nans == False:
        # Now we calculate VP and SD
        lvpsd = calculate_sd_vpd(lweather)
        lweather_all = insert_in_order(lweather, lvpsd)

        lveg =  dicveg[(x, y)]
        llc = diclc[(x, y)]
        key = (x, y, date)
        dic[key] = lweather_all + lveg + llc
        return dic[key]
    else:
        return None

def loadMask():
    path_mask = r"/home/irene/PycharmProjects/NL_predictors/data/mask.csv"
    mask = np.loadtxt(path_mask, delimiter=";")
    return mask

def loadMaskLandCover():
    zeros = np.zeros((350, 300))
    # path_mask =  r"D:\GeoData\workspaceimg\Special\Paper2\tifs\NL_LGN_1km_Loofd_Naald_Gras.tif"
    # path_to_tif = r"D:\GeoData\the_mask.csv"
    path_mask = r"/home/irene/PycharmProjects/NL_predictors/data/NL_LGN_1km_Loofd_Naald_Gras.tif"

    tif = gdal.Open(path_mask)
    mask = np.array(tif.GetRasterBand(1).ReadAsArray())

    mask[np.isnan(mask)] = -99
    mask[np.isneginf(mask)] = -99
    mask[mask<0] = -99
    mask[mask==0] = -99
    return mask

def generate_dates():
    start_date = datetime.datetime(2010, 1, 1)
    end_date = datetime.datetime(2011, 1, 1)
    delta = end_date - start_date
    dates = [start_date + datetime.timedelta(days=i) for i in range(delta.days)]
    return dates


################
# Main Program #
################

l = []
intify = []
counter = 1
writeHeaders = False

rows = range(0, 350)
cols = range(0, 300)



#dates = [
         # datetime.datetime(2014, 3, 1)
         # datetime.datetime(2014, 4, 1),
         # datetime.datetime(2014, 5, 1),
         # datetime.datetime(2014, 6, 1),
         # datetime.datetime(2014, 7, 1)
         # datetime.datetime(2014, 2, 1),
         # datetime.datetime(2014, 1, 1),
         # datetime.datetime(2014, 8, 1),
         # datetime.datetime(2014, 9, 1),
         # datetime.datetime(2014, 10, 1),
         # datetime.datetime(2014, 11, 1),
         # datetime.datetime(2014, 12, 1)
         # datetime.datetime(2013, 1, 1),
         # datetime.datetime(2013, 2, 1),
         # datetime.datetime(2013, 3, 1),
         # datetime.datetime(2013, 4, 1),
         # datetime.datetime(2013, 5, 1),
         # datetime.datetime(2013, 6, 1),
         # datetime.datetime(2013, 7, 1),
         # datetime.datetime(2013, 8, 1),
         # datetime.datetime(2013, 9, 1),
         # datetime.datetime(2013, 10, 1),
         # datetime.datetime(2013, 11, 1),
         # datetime.datetime(2013, 12, 1)

#         ]

gendates = generate_dates()

dicveg = loadVegetation()
diclc = loadLandCover()
mask_nl = loadMask()
mask_landcover = loadMaskLandCover()

# plt.imshow(mask_landcover, interpolation="None")
# plt.show()

inside = 0
written = 0
failed = 0

here = np.zeros((350, 300))

for date in gendates:
    print("Creating table for date: ", date)
    counter = 0
    formatted = path_out.format(date.year, date.month, date.day)
    with open(formatted, "w", newline="") as w:
        writer = csv.writer(w, delimiter=";")
        for x, y in itertools.product(rows, cols):
            if mask_landcover[x, y] in [11, 12, 20, 22, 23, 28, 45]:
                here[x, y] = 1
                inside+=1
                values = extract(x, y, date, dicveg, diclc)
                if values != None:
                    newrow = [counter, x, y] + values
                    if writeHeaders == False:
                        headers = ["rowid", "x", "y"] + cols_weather + cols_veg + cols_lc
                        writer.writerow(headers)
                        writeHeaders = True
                    roundedrow = np.round(np.array(newrow), decimals=2).tolist()
                    writer.writerow(roundedrow)
                    written+=1
                else:
                    failed += 1
            counter += 1

    print("Counter: ", counter)
    print("Inside: ", inside)
    print("Written: ", written)
    print("Failed: ", failed)

