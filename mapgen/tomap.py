import csv
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

    plt.imshow(here, interpolation="None")
    plt.show()
    return np.round(placed,decimals=2)



def write_tif(m, path):
    tif_template = gdal.Open("E:/RSData/KNMI/yearly/tmax/2014_tmax.tif")
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

path_mask = r"D:/Data/mask/mask_LGN_v2.csv"

mask = np.loadtxt(path_mask, delimiter=";")

path_pred_rf = r"D:\GeoData\workspaceimg\Special\IJGIS\predictions_v4\RF_NL_All_Predictors_LC_1_8_2014.csv"

path_out_rf = r"D:\GeoData\workspaceimg\Special\IJGIS\tifs\v4\RF_NL_All_Predictors_LC_1_8_2014.tif"

pred_rf = np.loadtxt(path_pred_rf, delimiter=";")

placed_rf = place_pixel(pred_rf)

write_tif(placed_rf, path_out_rf)
