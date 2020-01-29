import csv
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from evaluate import evaluate
import matplotlib.pyplot as plt
from collections import defaultdict

def select_columns(m, combination):
    return m[:, combination]

def descaling(target, descale):
    minimum = 0
    maximum = descale[2]
    X_scaled = np.round(target * (maximum - minimum) + minimum)
    return X_scaled

def getCombination_gp(headers_list):
    # return tuple(range(0, 102)) # + tuple([1327]) # + tuple(range(1327,1337))
    return (0,) + tuple(range(1, 102))
    # return tuple(range(0, len(headers_list)))

def getCombination_gp_temporal(headers_list):
    return (0,) + tuple(range(1, 107))


def getCombination_btu(headers_list):
    combinations = []
    # Here the fixed part contains "vegetation" and "landcover"
    # and the dynamic part will contain a chunk of the X-days predictors,
    # organized by the number of days prior to a date (eg tmin-5, tmax-5, prec-5, ev-5)
    tickcounts = (0,)
    combi_veg = (93, 94, 95, 96, 97, 98)
    combi_landcover = (99, 100, 101)
    for i in range(0, 11):
        chunk = (16+(7*i), 17+(7*i), 18+(7*i), 19+(7*i), 20+(7*i), 21+(7*i), 22+(7*i))
        a_combi = tickcounts + chunk + combi_veg + combi_landcover
        combinations.append(a_combi)
    return combinations

def getCombination_btu_temporal(headers_list):
    combinations = []
    # Here the fixed part contains "vegetation" and "landcover"
    # and the dynamic part will contain a chunk of the X-days predictors,
    # organized by the number of days prior to a date (eg tmin-5, tmax-5, prec-5, ev-5)
    tickcounts = (0,)
    combi_veg = (93, 94, 95, 96, 97, 98)
    combi_landcover = (99, 100, 101)
    combi_temporal = (102, 103, 104, 105, 106)
    for i in range(0, 11):
        chunk = (16+(7*i), 17+(7*i), 18+(7*i), 19+(7*i), 20+(7*i), 21+(7*i), 22+(7*i))
        a_combi = tickcounts + chunk + combi_veg + combi_landcover + combi_temporal
        combinations.append(a_combi)
    return combinations


def getCombination_mapping(headers_list):
    tickcounts = (0,)
    combi_weather = tuple(range(16, 93)) # before it was a 51, 62
    combi_veg = (93, 94, 95, 96, 97, 98)
    combi_landcover = (99, 100, 101)
    combination_fixed = tickcounts + combi_weather + combi_veg + combi_landcover
    combination = combination_fixed
    return combination

def getCombination_mapping_temporal(headers_list):
    tickcounts = (0,)
    combi_weather = tuple(range(16, 93)) # before it was a 51, 62
    combi_veg = (93, 94, 95, 96, 97, 98)
    combi_landcover = (99, 100, 101)
    combi_temporal = (102, 103, 104, 105, 106)
    combination_fixed = tickcounts + combi_weather + combi_veg + combi_landcover + combi_temporal
    combination = combination_fixed
    return combination

def encode_habitat(m):
    newcols = []
    cols_habitat = (2, 3, 4)
    mselcols = select_columns(m, cols_habitat)
    for col in mselcols.T:
        le = preprocessing.LabelEncoder()
        le.fit(col)
        newcol = le.transform(col)
        newcols.append(newcol)
    i = 0
    for col in newcols:
        for idx in cols_habitat:
            m[i, idx] = col[i]
            i+=1
        i = 0
    return m

def normalize_target_zscore(observations):
    dic = defaultdict(list)
    for i in range(len(observations)):
        key = observations[i][1]
        dic[key].append(observations[i])

    dic_mean = defaultdict(list)
    for key in sorted(dic.keys()):
        for i in range(len(dic[key])):
            tick_count = dic[key][i][3]
            dic_mean[key].append(tick_count)

    dic_summary = defaultdict(list)
    for key in dic_mean:
        floatify = [float(item) for item in dic_mean[key]]
        mean_key = np.mean(floatify)
        std_key = np.std(floatify)
        dic_summary[key] = (mean_key, std_key)
        print(key, mean_key, std_key)

    z_scores = []
    for row in observations:
        where = row[1]
        mean, std = dic_summary[where]
        tick_count = float(row[3])
        score = np.divide(np.subtract(tick_count, mean), std)
        z_scores.append(score)

    return np.array(z_scores), dic_summary

def scale_NL(X):
    cols = []
    counter = 0
    descale = []
    descale_target = None
    continuous_features = list(range(0, 86))
    for feature in X.T:
        if counter in continuous_features:
            minimum = feature.min(axis=0)
            maximum = feature.max(axis=0)
            col_std = np.divide((feature - minimum), (maximum - minimum))
            cols.append(col_std)
            descale.append((counter, minimum, maximum))
        else:
            cols.append(feature)
        counter += 1
    X_std = np.array(cols)
    return X_std.T, descale, descale_target


def scale(X, all_observations):
    cols = []
    counter = 0
    descale = []
    descale_target = None

    # Also tried normalizing Vegetation indices and nothing
    continuous_features = [0] + list(range(16, 100)) # + [104, 105, 106, 108, 110, 112]
    # z_score, descale_target = normalize_target_zscore(all_observations)
    # cols.append(z_score)
    # continuous_features = list(range(17, 93)) # + [104, 105, 106, 108, 110, 112]
    # continuous_features = list(range(0, 101))
    for feature in X.T:
        if counter in continuous_features:
            minimum = feature.min(axis=0)
            maximum = feature.max(axis=0)
            col_std = np.divide((feature - minimum), (maximum - minimum))
            cols.append(col_std)
            descale.append((counter, minimum, maximum))
        else:
            cols.append(feature)
        counter += 1
    X_std = np.array(cols)
    return X_std.T, descale, descale_target

def explore(m):
    tar = m[:, 0]
    tmin = m[:, 16]
    tmax = m[:, 17]
    prec = m[:, 18]
    ev = m[:, 19]
    rh = m[:, 20]
    sd = m[:, 21]
    vp = m[:, 22]

    plt.subplot(2, 4, 1)
    plt.hist(tar, bins=10)

    plt.subplot(2, 4, 2)
    plt.hist(tmin, bins=10)

    plt.subplot(2, 4, 3)
    plt.hist(tmax, bins=10)

    plt.subplot(2, 4, 4)
    plt.hist(prec, bins=10)

    plt.subplot(2, 4, 5)
    plt.hist(ev, bins=10)

    plt.subplot(2, 4, 6)
    plt.hist(rh, bins=10)

    plt.subplot(2, 4, 7)
    plt.hist(sd, bins=10)

    plt.subplot(2, 4, 8)
    plt.hist(vp, bins=10)

    plt.show()


def load_stuff_per_site(path_in, name, experiment=0):
    # banned = ["KwadeHoek_1", "KwadeHoek_2", "Appelscha_1", "Appelscha_2", "Bilthoven_1", "Bilthoven_2", "Dronten_1", "Dronten_2", "Schiermonnikoog_1", "Schiermonnikoog_2", "Vaals_1", "Vaals_2"]
    banned = []
    all_observations = []
    sites_and_dates = []
    site_one = []
    site_two = []
    name1 = name + "_1"
    name2 = name + "_2"
    with open(path_in, "r", newline="") as r:
        headers = next(r)
        headers_list = headers.split(";")[3:]
        reader = csv.reader(r, delimiter=";")
        for row in reader:
            if row[1] not in banned:
                date = datetime.datetime.strptime(row[2], "%Y-%m-%d").date()
                tick_count_in_site = float(row[3])
                if tick_count_in_site <= 100:
                    all_observations.append(row)
                    floatify = [float(item) for item in row[3:]]
                    # floatified.append(floatify)
                    sites_and_dates.append((row[1], date))
                if row[1] == name1:
                    site_one.append(floatify)
                elif row[1] == name2:
                    site_two.append(floatify)

    if len(site_one) == 0:
        return [None] * 7

    matone = np.array(site_one)
    mattwo = np.array(site_two)
    print(matone.shape, mattwo.shape)
    marrayone = encode_habitat(matone.astype(dtype="float32", casting="same_kind"))
    marraytwo = encode_habitat(mattwo.astype(dtype="float32", casting="same_kind"))
    combination = getCombination_gp(headers_list)

    mscaled1, descale1, skip = scale(marrayone, all_observations)
    mscaled2, descale2, skip = scale(marraytwo, all_observations)
    mrounded1 = np.round(mscaled1, decimals=2)
    mrounded2 = np.round(mscaled2, decimals=2)

    if experiment in [0, 2]:
        # print("\t\t\tColumns to be used: ", [headers_list[item] for item in combination])
        msel1 = select_columns(mrounded1, combination)
        msel2 = select_columns(mrounded2, combination)

    return msel1, msel2, all_observations, headers_list, combination, descale1, descale2

def load_pretty_sites(path_in, experiment=0):
    l25 = []
    banned = ["KwadeHoek_1", "KwadeHoek_2", "Appelscha_1", "Appelscha_2", "Bilthoven_1", "Bilthoven_2", "Dronten_1", "Dronten_2", "Schiermonnikoog_1", "Schiermonnikoog_2", "Vaals_1", "Vaals_2"]
    all_observations = []
    floatified = []
    sites_and_dates = []
    peaks = 0

    with open(path_in, "r", newline="") as r:
        headers = next(r)
        headers_list = headers.split(";")[3:]
        reader = csv.reader(r, delimiter=";")
        for row in reader:
            if row[1] not in banned:
                # if row[1].split("_")[1] == "2":
                date = datetime.datetime.strptime(row[2], "%Y-%m-%d").date()
                tick_count_in_site = float(row[3])
                if tick_count_in_site <= 100:
                    all_observations.append(row)
                    floatify = [float(item) for item in row[3:]]
                    floatified.append(floatify)
                    sites_and_dates.append((row[1], date))
                else:
                    peaks+=1

    # explore(np.array(l25))
    # Data are read, we go for the preprocessing
    mat = np.array(floatified)
    marray = encode_habitat(mat.astype(dtype="float32", casting="same_kind"))
    print("\nRead raw matrix: ", marray.shape)

    # Column selection
    if experiment == 0:
        print("Selected combination for general performance")
        combination = getCombination_gp(headers_list)
    elif experiment == 1:
        print("Selected combination for best temporal unit")
        combination = getCombination_btu(headers_list)
        print(combination)
    elif experiment == 2:
        print("Selected combination for mapping")
        combination = getCombination_mapping(headers_list)

    print("Scaling matrix and selecting columns")
    # the_target = marray[:, 0]
    # plt.hist(the_target, bins=20)
    # plt.show()
    mscaled, descale, descale_target = scale(marray, all_observations)
    mrounded = np.round(mscaled, decimals=2)
    print(mrounded.shape)
    if experiment in [0, 2]:
        # print("\t\t\tColumns to be used: ", [headers_list[item] for item in combination])
        msel = select_columns(mrounded, combination)
    else:
        msel = mrounded
    return msel, all_observations, headers_list, combination, descale, descale_target


def load_stuff(path_in, experiment=0):
    l25 = []
    # banned = ["Wassenaar_1", "Wassenaar_2", "Vaals_1", "Schiermonnikoog_1", "Schiermonnikoog_2", "Montferland_2", "KwadeHoek_2", "Appelscha_1", "Appelscha_2", "Bilthoven_1", "Bilthoven_2"]
    # banned = ["KwadeHoek_1", "KwadeHoek_2", "Appelscha_1", "Appelscha_2", "Bilthoven_1", "Bilthoven_2", "Dronten_1", "Dronten_2", "Schiermonnikoog_1", "Schiermonnikoog_2", "Vaals_1", "Vaals_2"]
    banned = []
    all_observations = []
    floatified = []
    sites_and_dates = []
    peaks = 0

    with open(path_in, "r", newline="") as r:
        headers = next(r)
        headers_list = headers.split(";")[3:]
        reader = csv.reader(r, delimiter=";")
        for row in reader:
            if row[1] not in banned:
                # if row[1].split("_")[1] == "2":
                date = datetime.datetime.strptime(row[2], "%Y-%m-%d").date()
                tick_count_in_site = float(row[3])
                if tick_count_in_site <= 1000:
                    all_observations.append(row)
                    floatify = [float(item) for item in row[3:]]
                    floatified.append(floatify)
                    sites_and_dates.append((row[1], date))
                else:
                    peaks+=1

    # explore(np.array(l25))
    # Data are read, we go for the preprocessing
    mat = np.array(floatified)
    marray = encode_habitat(mat.astype(dtype="float32", casting="same_kind"))
    print("\nRead raw matrix: ", marray.shape)

    # Column selection
    if experiment == 0:
        print("Selected combination for general performance!!")
        combination = getCombination_gp(headers_list)
    elif experiment == 1:
        print("Selected combination for best temporal unit")
        combination = getCombination_btu(headers_list)
        print(combination)
    elif experiment == 2:
        print("Selected combination for mapping")
        combination = getCombination_mapping(headers_list)

    print("Scaling matrix and selecting columns")
    # the_target = marray[:, 0]
    # plt.hist(the_target, bins=20)
    # plt.show()
    mscaled, descale, descale_target = scale(marray, all_observations)
    mrounded = np.round(mscaled, decimals=2)
    print(mrounded.shape)
    if experiment in [0, 2]:
        # print("\t\t\tColumns to be used: ", [headers_list[item] for item in combination])
        msel = select_columns(mrounded, combination)
    else:
        msel = mrounded
    return msel, all_observations, headers_list, combination, descale, descale_target
    return msel, all_observations, headers_list, combination, descale, descale_target

def show_model_evaluation(ytest, y_pred):
    evaluation = evaluate(ytest, y_pred)
    # rmse = np.sqrt(mean_squared_error(ytest, y_pred))
    rmse = evaluation[-4]
    rmses = evaluation[-3]
    rmseu = evaluation[-2]
    the_mean = np.mean(ytest)
    the_mean_pred = np.mean(y_pred)
    nrmse = np.divide(rmse, the_mean)
    # mae = mean_absolute_error(ytest, y_pred)
    mae = evaluation[-6]
    r2 = r2_score(ytest, y_pred)
    pack = [rmse, rmses, rmseu, nrmse, mae, r2, the_mean, the_mean_pred]
    return pack


def load_stuff_temporal(path_in, experiment=0):
    banned = []
    all_observations = []
    floatified = []
    sites_and_dates = []
    peaks = 0

    with open(path_in, "r", newline="") as r:
        headers = next(r)
        headers_list = headers.split(";")[3:]
        reader = csv.reader(r, delimiter=";")
        for row in reader:
            if row[1] not in banned:
                # if row[1].split("_")[1] == "2":
                date = datetime.datetime.strptime(row[2], "%Y-%m-%d").date()
                tick_count_in_site = float(row[3])
                if tick_count_in_site <= 1000:
                    all_observations.append(row)
                    floatify = [float(item) for item in row[3:]]
                    floatified.append(floatify)
                    sites_and_dates.append((row[1], date))
                else:
                    peaks+=1

    # explore(np.array(l25))
    # Data are read, we go for the preprocessing
    mat = np.array(floatified)
    marray = encode_habitat(mat.astype(dtype="float32", casting="same_kind"))
    print("\nRead raw matrix: ", marray.shape)

    # Column selection
    if experiment == 0:
        print("Selected combination for general performance!!")
        combination = getCombination_gp_temporal(headers_list)
    elif experiment == 1:
        print("Selected combination for best temporal unit")
        combination = getCombination_btu_temporal(headers_list)
        print(combination)
    elif experiment == 2:
        print("Selected combination for mapping")
        combination = getCombination_mapping_temporal(headers_list)

    print("Scaling matrix and selecting columns")
    # the_target = marray[:, 0]
    # plt.hist(the_target, bins=20)
    # plt.show()
    mscaled, descale, descale_target = scale(marray, all_observations)
    mrounded = np.round(mscaled, decimals=2)
    print(mrounded.shape)
    if experiment in [0, 2]:
        # print("\t\t\tColumns to be used: ", [headers_list[item] for item in combination])
        msel = select_columns(mrounded, combination)
    else:
        msel = mrounded
    return msel, all_observations, headers_list, combination, descale, descale_target
    return msel, all_observations, headers_list, combination, descale, descale_target