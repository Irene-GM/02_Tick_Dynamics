import csv
import numpy as np
import gdal
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from collections import defaultdict


def count_zeros_ones(arr):
    zeros = 0
    for elem in arr.tolist():
        if elem[0]==0:
            zeros += 1
    ones = len(arr) - zeros
    return zeros, ones

def stamp_totals(zeros, ones):
    n = zeros + ones
    percent_z = np.round(np.divide(zeros * 100, n), decimals=1)
    percent_o = np.round(np.divide(ones * 100, n), decimals=1)
    print("Total Zeros: ", zeros, "\t{0}%".format(percent_z))
    print("Total Ones: ", ones, "\t{0}%".format(percent_o))
    print("Total Samples: ", n)

def stamp_metrics(ytrain, ytest):

    pack = [ytrain.reshape(-1, 1), ytest.reshape(-1, 1)]
    labels = ["Ytrain: ", "Ytest: "]
    total_zeros = 0
    total_ones = 0
    i = 0
    for item in pack:
        arr = item
        zeros, ones = count_zeros_ones(arr)
        total_zeros += zeros
        total_ones += ones
        print(labels[i])
        print("\t\t Zeros: ", zeros)
        print("\t\t Ones: ", ones)
        print("\t\t Shape: ", arr.shape)
        print()
        i+=1

    stamp_totals(total_zeros, total_ones)

def select_columns(m, combination):
    return m[:, combination]

def printing(rf, xtest, ytest, pred_rf):
    print("\nMetrics")
    print("-"*40)
    rf_sco = round(rf.score(xtest, ytest),2)
    f1_sco = round(f1_score(ytest, pred_rf, average="macro"), 2)
    prec_sco = round(precision_score(ytest, pred_rf, average="macro"), 2)
    recall_sco = round(recall_score(ytest, pred_rf, average="macro"), 2)
    print("RF Score: \t", rf_sco)
    print("F1 Score: \t", f1_sco)
    print("Precision: \t", prec_sco)
    print("Recall: \t", recall_sco)
    print()
    print("Confusion Matrix")
    print(confusion_matrix(ytest, pred_rf))


def write_tif(m, path):
    tif_template = gdal.Open("F:/RSData/KNMI/yearly/tmax/2014_tmax.tif")
    rows = tif_template.RasterXSize
    cols = tif_template.RasterYSize

    # Get the origin coordinates for the tif file
    geotransform = tif_template.GetGeoTransform()
    outDs = tif_template.GetDriver().Create(path, rows, cols, 1, gdal.GDT_Float32)
    outBand = outDs.GetRasterBand(1)

    # write the data
    outDs.GetRasterBand(1).WriteArray(m)

    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    outBand.SetNoDataValue(-99)

    # georeference the image and set the projection
    outDs.SetGeoTransform(geotransform)
    outDs.SetProjection(tif_template.GetProjection())
    outDs = None
    outBand = None

def place_pixel(m):
    placed = np.multiply(np.ones((350, 300)), -99)
    for position in m:
        row = position[1]
        col = position[2]
        value = position[3]
        placed[row, col] = value
    return placed

################
# Main program #
################


path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_raw_binarized.csv"
path_tables = r"D:\GeoData\workspaceimg\Special\Paper2\testing_tables_v3\\"
path_out_pred = r"M:\Documents\workspace\Special\IJGIS\data\zero_detector\pred_{0}.csv"
path_out_img = r"M:\Documents\workspace\Special\IJGIS\data\zero_detector\{0}.tif"

l = []
combination = (0,) + tuple(range(16, 102))

with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    next(reader)
    for row in reader:
        floatify = [float(item) for item in row[3:]]
        l.append(floatify)

mraw = np.array(l)
m = select_columns(mraw, combination)
maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))

# Read and prepare data
xtrain = m[:lim, 1:]
ytrain = m[:lim, 0]
xtest = m[lim:, 1:]
ytest = m[lim:, 0]

print("xtrain: {0} \tytrain: {1} \txtest: {2} \tytest: {3}".format(xtrain.shape, ytrain.shape, xtest.shape, ytest.shape))

print("ytrain 0 to 5", ytrain[0:5])
print("ytest 0 to 5", ytest[0:5])

stamp_metrics(ytrain, ytest)

rf = RandomForestClassifier(n_estimators=300,
                            max_features=50,
                            max_depth=10,
                            min_samples_leaf=10,
                            criterion="entropy",
                            min_samples_split=1,
                            bootstrap=True)

rf.fit(xtrain, ytrain)
pred_rf = rf.predict(xtest)

printing(rf, xtest, ytest, pred_rf)

for root, dirs, files in os.walk(path_tables):
    for file in files:
        filepath = path_tables + file
        print("\nReading: ", file)
        with open(filepath, "r", newline="") as r:
            reader = csv.reader(r, delimiter=";")
            next(reader)
            dic_positions = defaultdict(list)
            l = []
            order = []
            for row in reader:
                key = float(row[0])
                r = float(row[1])
                c = float(row[2])
                dic_positions[key] = [r, c]
                order.append(key)
                floatify = [float(item) for item in row[3:]]
                l.append(floatify)
            mtest = np.array(l)
            pred_file = rf.predict(mtest)

            print("\tZeros predicted: ", pred_file.tolist().count(0))
            print("\tOnes predicted: ", pred_file.tolist().count(1))

            i = 0
            idx = []
            for rowid in order:
                r, c = dic_positions[rowid]
                value = pred_file[i]
                newrow = [rowid, r, c, value]
                idx.append(newrow)
                i += 1

            midx = np.array(idx)
            outname = path_out_pred.format(file.split(".")[0])
            outname_tif = path_out_img.format(file.split(".")[0])
            np.savetxt(outname, midx, delimiter=";", fmt="%d")
            placed_pixels = place_pixel(midx)
            write_tif(placed_pixels, outname_tif)
