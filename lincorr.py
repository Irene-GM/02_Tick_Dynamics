import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sb
from sklearn.metrics import r2_score
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter


def select_columns(m, combination):
    return m[:, combination]


################
# Main program #
################

sb.set_style("dark")

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_raw.csv"

labels = ["tmin", "tmax", "prec", "ev", "rh", "sd", "vp", "min_ndvi", "range_ndvi", "min_evi", "range_evi", "min_ndwi", "range_ndwi"]

combination = (0,) + tuple(range(16, 23)) + tuple(range(93, 99))
# combination = (0,) + tuple(range(63, 82)) + tuple(range(93, 99))

l = []
with open(path_in, "r", newline="") as r:
    reader = csv.reader(r, delimiter=";")
    next(reader)
    for row in reader:
        floatify = [float(item) for item in row[3:]]
        l.append(floatify)

mraw = np.array(l)
print(mraw.shape)
m = select_columns(mraw, combination)

r2_lbl = " (R2={0})"

target = m[:, 0]
for i in range(1, 14):

    ax = plt.subplot(2, 7, i)
    plt.subplots_adjust(hspace=.2)
    plt.subplots_adjust(wspace=.3)
    plt.suptitle("Linear fitting of AQT with each weather and vegetation features", size=18)
    plt.ylim(0, 150)
    plt.xticks(rotation=30)
    if labels[i-1] == "prec":
        plt.xlim(-0.5, 20)

    variable = m[:, i]

    sb.regplot(x=variable, y=target, color="grey", fit_reg=True, ci=None, line_kws={'color':'black'})

    slope, intercept, r_value, p_value, std_err = stats.linregress(variable, target)
    plt.title(labels[i-1].upper() + r2_lbl.format(np.round(r_value**2, decimals=2)), size=14)

    plt.ylabel("AQT")
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.grid(b=True, which='major', color='#A6A6A6', linewidth=0.5)
    ax.grid(b=True, which='major', color='#A6A6A6', linewidth=0.5)
    i += 1

plt.show()




