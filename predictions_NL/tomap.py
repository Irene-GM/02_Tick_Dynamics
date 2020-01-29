import os
import numpy as np
import gdal
import matplotlib.pyplot as plt

def place(m, mask):
    placed = np.multiply(np.ones((350, 300)), -99)
    k = 0
    for i in range(0, 350):
        for j in range(0, 300):
            if mask[i, j] == 1:
                prediction = m[k, 0]
                placed[i, j] = prediction
                k += 1
    return np.round(placed, decimals=2)

def place_pixel(m):
    here = np.zeros((350, 300))
    placed = np.multiply(np.ones((350, 300)), -99)
    for position in m:
        row = position[1]
        col = position[2]
        here[row, col] = 1
        value = position[3]
        placed[row, col] = value
    return np.round(placed,decimals=2)



def write_tif(m, path):
    tif_template = gdal.Open("/datasets/KNMI/yearly/tmax/2014_tmax.tif")
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



################
# Main program #
################

path_mask = r"/home/irene/PycharmProjects/NL_predictors/data/mask_LGN_v2.csv"

mask = np.loadtxt(path_mask, delimiter=";")

version = 7

path_pred_rf = r"/home/irene/PycharmProjects/NL_predictors/data/versions/v8/predictions_v8/2014"

path_out_rf = r"/home/irene/PycharmProjects/NL_predictors/data/versions/v8/maps_v8/2014/"

template = "NL_Map_{0}_{1}.tif"

for subdir, dirs, files in os.walk(path_pred_rf.format(version)):
    for file in files:
        print("\nWorking with: \t", file)
        name = path_pred_rf.format(version) + "/" + file
        pred_rf = np.loadtxt(name, delimiter=";")
        placed_rf = place_pixel(pred_rf)
        new_name = template.format("RF", file[:-4])
        out_path = path_out_rf + new_name
        write_tif(placed_rf, out_path)