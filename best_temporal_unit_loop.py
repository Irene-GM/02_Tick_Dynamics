import sys
sys.path.append(r'M:\Documents\workspace\Special\IJGIS\new')
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils import *
import seaborn as sb
import matplotlib.pyplot as plt
from collections import defaultdict
from operator import itemgetter
import datetime


def show_error_plots(rmse_list, rmses_list, rmseu_list, nrmse_list, labels):
    path_fig = r"\\ad.utwente.nl\home\garciamartii\Documents\PhD\Papers\Journals\02_IJGIS\images\print_v5\{0}"
    name_fig = "Figure_05_Performance_Time_Scales.png"
    s = 30
    plt.clf()
    plt.close()
    xlinspace = np.linspace(0, 10, 11)

    # plt.suptitle("Performance of RF at multiple time scales", size=36)

    plt.subplots_adjust(hspace=.5, wspace=.3)
    ax1 = plt.subplot(2, 3, 1)
    plt.plot(xlinspace, nrmse_list, "-", label="NRMSE", linewidth=2, color="darkblue")
    # plt.title("Evolution of NRMSE", size=s)
    plt.xticks(xlinspace, labels, size='medium', rotation=70, fontsize = 22)
    plt.xlabel('Temporal aggregation', size=s)
    plt.ylabel('Normalized RMSE', size=s)
    ax1.grid(b=True, which='major', color='#A6A6A6', linewidth=0.5)
    ax1.grid(b=True, which='major', color='#A6A6A6', linewidth=0.5)

    ax2 = plt.subplot(2, 3, 2)
    plt.plot(xlinspace, rmse_list, "-", color="darkblue", label="RMSE", linewidth=3)
    plt.plot(xlinspace, rmses_list, "-", color="darkgreen", label="RMSEs", linewidth=2)
    plt.plot(xlinspace, rmseu_list, "-", color="darkgray", label="RMSEu", linewidth=2)
    # plt.title("Evolution of RMSE", size=s)
    plt.xlabel('Temporal aggregation', size=s)
    plt.ylabel('RMSE', size=s)
    plt.xticks(xlinspace, labels, size='medium', rotation=70, fontsize = 22)
    ax2.grid(b=True, which='major', color='#A6A6A6', linewidth=0.5)
    ax2.grid(b=True, which='major', color='#A6A6A6', linewidth=0.5)
    ax2.legend(bbox_to_anchor=(1.5, 0.0), loc='lower right', borderaxespad=0., prop={'size':20})

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    fig = plt.gcf()
    plt.show()
    fig.savefig(path_fig.format(name_fig), format="png", dpi=300)


def show_feature_importances(rf, combination, headers, lim):
    cols = [headers[item] for item in combination[1:]] # Skip label tick count
    i = 1
    print("\nFeature Importances")
    print("-"*40)
    for item in list(reversed(sorted(zip(cols, rf.feature_importances_), key=lambda x: x[1])))[:lim]:
        score = np.round(item[1] * 100, decimals=2)
        dicfirf[item[0]].append(score)
        print(i, ")\t", item[0], "\t\t", score, "%")
        i += 1
    print(dicfirf)

def show_avg_fi(dic):
    l = []
    for key in dic.keys():
        mean = np.mean(dic[key])
        l.append((key, mean))

    lsort = list(reversed(sorted(l, key=itemgetter(1))))
    return lsort[0:10]


def bound_target(path):
    l = []
    k = 1
    idx = np.genfromtxt(path_in, delimiter=";", usecols=range(0,4), dtype=str, skip_header=1)
    dic = defaultdict(list)
    for row in idx:
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
        value = float(row[3])
        dic[date.month].append(value)
    data = [dic[i] for i in range(1, 13)]
    ax = plt.boxplot(data)
    dicwhiskers = defaultdict(tuple)
    for i in range(2, 25, 2):
        lower_whisker = ax['whiskers'][i-2].get_ydata()
        upper_whisker = ax['whiskers'][i-1].get_ydata()
        dicwhiskers[k] = [lower_whisker, upper_whisker]
        k += 1

    for row in idx:
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
        value = float(row[3])
        low_whisker = dicwhiskers[date.month][0]
        upp_whisker = dicwhiskers[date.month][1]
        low_bound = low_whisker[0]
        upp_bound = upp_whisker[1]
        Y_value_bounded = np.divide((value - low_bound), (upp_bound - low_bound))
        l.append(Y_value_bounded)

    return [np.array(l), dicwhiskers, idx]

def bounded_descaling(ypred, dicwhiskers, idx, lim):
    i = 0
    l = []
    test_metadata = idx[lim:, :]
    for row in test_metadata:
        date = datetime.datetime.strptime(row[2], "%Y-%m-%d")
        low_whisker = dicwhiskers[date.month][0]
        upp_whisker = dicwhiskers[date.month][1]
        low_bound = low_whisker[0]
        upp_bound = upp_whisker[1]
        Y_value_bounded = np.round(ypred[i] * (upp_bound - low_bound) + low_bound)
        l.append(Y_value_bounded)
        i += 1
    return np.array(l)



################
# Main program #
################

dicfirf = defaultdict(list)

path_in = r"M:\Documents\workspace\Special\IJGIS\data\new\random_FS_nymphs_with_zeros_savitzky_golay.csv"

m, all_observations, headers_list, combinations, descale, descale_target = load_stuff(path_in, experiment=1)

target_limits = descale[0]
maxlim = m.shape[0]
lim = int(np.divide(maxlim * 70, 100))
origY = descaling(m[:,0], target_limits)

nrmse_list, rmse_list, rmses_list, rmseu_list, r2_list, mae_list = ([] for i in range(6))

labels = ["Xdays-1", "Xdays-2", "Xdays-3", "Xdays-4", "Xdays-5", "Xdays-6", "Xdays-7", "Xdays-14", "Xdays-30", "Xdays-90", "Xdays-365"]

ldic, lrf, lgbr, lrmse, lrmses, lrmseu, lnrmse, partial_rmse, partial_rmses, partial_rmseu, partial_nrmse = ([] for i in range(11))

target_bound, whiskers, idx = bound_target(path_in)

for combination in combinations:
    msel = select_columns(m, combination)
    # xtrain = msel[:lim, 1:]
    # ytrain = msel[:lim, 0]
    # xtest = msel[lim:, 1:]
    # ytest = msel[lim:, 0]
    xtrain = msel[:lim, 1:]
    ytrain = target_bound[:lim]
    xtest = msel[lim:, 1:]
    ytest = target_bound[lim:]
    old_mean = np.mean(origY[lim:])
    print("old mean: ", old_mean)

    print("Training...\n")

    # Now we go for the modelling
    for i in range(0, 10):
        rf = RandomForestRegressor(n_estimators=500, n_jobs=4, max_features="auto", oob_score=False, bootstrap=True, criterion="mse", max_leaf_nodes=None, max_depth=None, min_samples_leaf=1, min_samples_split=1, warm_start=False)
        # gbr = GradientBoostingRegressor(loss='huber', alpha=0.95, learning_rate=.05, max_features=None, max_leaf_nodes=None, criterion="mse", min_samples_split=2, min_samples_leaf=3, n_estimators=400, presort=False, max_depth=5)
        rf.fit(xtrain, ytrain)
        # gbr.fit(xtrain, ytrain)
        pred_rf = rf.predict(xtest)
        # pred_gbr = gbr.predict(xtest)

        # WITHOUT THE Z-SCORES!!!!
        # pred_rf_desc = np.round(descaling(pred_rf, target_limits), decimals=0)
        # pred_gbr_desc = np.round(descaling(pred_gbr, target_limits), decimals=0)
        # ytest_desc = np.round(descaling(ytest, target_limits), decimals=0)

        pred_rf_desc = bounded_descaling(pred_rf, whiskers, idx, lim)
        ytest_desc = bounded_descaling(ytest, whiskers, idx, lim)

        show_feature_importances(rf, combination, headers_list, 10)
        # show_feature_importances(gbr, combination, headers_list, 5)

        rfrmse, rfrmses, rfrmseu, rfnrmse, rfmae, rfr2, rfmean, rfmean_pred = show_model_evaluation(ytest_desc, pred_rf_desc)
        # gbrmse, gbrmses, gbrmseu, gbnrmse, gbmae, gbr2, gbmean, gbmean_pred = show_model_evaluation(ytest_desc, pred_gbr_desc)

        good_nrmse = np.divide(rfrmse, old_mean)

        # lrf.append((rf, pred_rf_desc, rfrmse, rfrmses, rfrmseu, rfnrmse, rfmae, rfr2, rfmean, rfmean_pred))
        # lgbr.append((gbr, pred_gbr_desc, gbrmse, gbrmses, gbrmseu, gbnrmse, gbmae, gbr2, gbmean, gbmean_pred))

        partial_rmse.append(rfrmse)
        partial_rmses.append(rfrmses)
        partial_rmseu.append(rfrmseu)
        partial_nrmse.append(good_nrmse)
        print("RMSE: ", rfrmse)
        print("Old mean: ", old_mean)
        print("NRMSE:", good_nrmse)
        print("")

    lrmse.append(np.mean(np.array(partial_rmse)))
    lrmses.append(np.mean(np.array(partial_rmses)))
    lrmseu.append(np.mean(np.array(partial_rmseu)))
    lnrmse.append(np.mean(np.array(partial_nrmse)))

    partial_rmse, partial_rmses, partial_rmseu, partial_nrmse = ([] for i in range(4))

    ldic.append(dicfirf)
    dicfirf = defaultdict(list)

show_error_plots(lrmse, lrmses, lrmseu, lnrmse, labels)
plt.show()

# for dic in ldic:
#     print(show_avg_fi(dic))



# NO ZEROS AND SAVITZKY-GOLAY
# [('ev-1', 29.064999999999998), ('rh-1', 12.062000000000001), ('tmax-1', 9.2570000000000014), ('tmin-1', 8.6070000000000011), ('LandCover_1km\r\n', 5.133)]
# [('ev-2', 27.282999999999998), ('rh-2', 11.216999999999999), ('tmin-2', 9.8270000000000017), ('tmax-2', 8.5920000000000023), ('prec-2', 5.516)]
# [('ev-3', 28.550000000000001), ('tmin-3', 11.254000000000001), ('rh-3', 9.1650000000000027), ('tmax-3', 8.2390000000000008), ('prec-3', 6.4712499999999995)]
# [('ev-4', 27.367000000000001), ('rh-4', 10.404999999999999), ('tmin-4', 9.6020000000000003), ('tmax-4', 7.8860000000000001), ('prec-4', 7.444)]
# [('ev-5', 24.456), ('rh-5', 11.108000000000001), ('tmin-5', 10.089), ('sd-5', 7.5587499999999999), ('tmax-5', 7.4337499999999999)]
# [('ev-6', 17.225000000000001), ('vp-6', 12.405000000000001), ('rh-6', 11.318000000000001), ('tmin-6', 10.539), ('tmax-6', 7.1950000000000003)]
# [('ev-7', 15.597), ('sd-7', 13.716999999999999), ('rh-7', 11.022), ('tmin-7', 10.099), ('prec-7', 8.0809999999999995)]
# [('rh-14', 25.728999999999996), ('ev-14', 12.833000000000002), ('tmin-14', 9.2129999999999992), ('prec-14', 8.907), ('tmax-14', 6.963000000000001)]
# [('rh-30', 27.488999999999997), ('ev-30', 10.334), ('tmin-30', 9.4149999999999991), ('prec-30', 8.8159999999999989), ('tmax-30', 7.0879999999999992)]
# [('rh-90', 22.880000000000003), ('tmin-90', 17.324000000000002), ('ev-90', 9.1170000000000009), ('prec-90', 8.9250000000000007), ('tmax-90', 6.4510000000000005)]
# [('rh-365', 19.660000000000004), ('tmax-365', 14.497), ('ev-365', 12.138), ('tmin-365', 12.117999999999999), ('prec-365', 10.876999999999999)]

# WITH ZEROS AND SAVITZKY-GOLAY
# [('ev-1', 27.643000000000001), ('tmax-1', 9.0600000000000005), ('rh-1', 8.8710000000000004), ('tmin-1', 8.7439999999999998), ('min_ndvi', 6.4510000000000005)]
# [('ev-2', 27.691000000000003), ('tmin-2', 9.1769999999999996), ('tmax-2', 8.8530000000000015), ('rh-2', 8.104000000000001), ('min_ndvi', 6.7889999999999997)]
# [('ev-3', 27.377000000000002), ('tmin-3', 9.9770000000000003), ('tmax-3', 8.5039999999999996), ('rh-3', 8.4559999999999995), ('min_ndvi', 6.7469999999999999)]
# [('ev-4', 27.972000000000001), ('tmin-4', 9.2300000000000004), ('tmax-4', 8.3509999999999991), ('rh-4', 7.6020000000000012), ('min_ndvi', 6.9399999999999995)]
# [('ev-5', 27.235000000000003), ('tmin-5', 8.9399999999999995), ('tmax-5', 7.819), ('rh-5', 7.694), ('min_ndvi', 6.9120000000000008)]
# [('ev-6', 23.494999999999997), ('tmin-6', 8.8240000000000016), ('tmax-6', 7.8179999999999996), ('rh-6', 7.5069999999999997), ('sd-6', 7.2877777777777784)]
# [('ev-7', 22.991999999999997), ('tmin-7', 8.4659999999999993), ('sd-7', 8.0287500000000005), ('rh-7', 7.9510000000000005), ('tmax-7', 7.5722222222222229)]
# [('ev-14', 15.597), ('rh-14', 14.559000000000001), ('tmin-14', 8.9139999999999979), ('prec-14', 7.9080000000000013), ('sd-14', 7.5709999999999997)]
# [('rh-30', 25.338000000000001), ('ev-30', 9.8499999999999979), ('tmin-30', 8.8089999999999993), ('prec-30', 8.1500000000000004), ('tmax-30', 6.604000000000001)]
# [('rh-90', 22.720000000000002), ('tmin-90', 14.843), ('prec-90', 8.282), ('ev-90', 7.8620000000000001), ('range_ndwi', 6.6440000000000001)]
# [('ev-365', 16.346999999999998), ('tmax-365', 15.097), ('prec-365', 12.178999999999998), ('rh-365', 12.059999999999999), ('tmin-365', 11.891)]