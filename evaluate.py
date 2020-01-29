import numpy as np

def evaluate(observations, predictions):
    obs = np.array(observations)
    pred = np.array(predictions)
    n = len(obs)
    sum_obs = sum(obs ** 2)
    sq_sum = sum((obs - pred) ** 2)

    # Mean of the observations and predictions
    mean_obs = np.mean(obs)
    mean_pred = np.mean(pred)

    # Standard deviation of the observations and predictions
    stdev_obs = np.std(obs)
    stdev_pred = np.std(pred)

    # Correlation coefficient
    # Returns a matrix like this:
    # [[1.0, 0.9923316965546289], [0.9923316965546289, 1.0]]
    correlation = np.corrcoef(obs, pred)

    # Intercept and slope of OLS regression
    #   Intercept is a matrix like this:
    #   array([[-0.02669013, -0.02264611], [-0.02264611, -0.02669013]])
    #   Slope is a matrix like this
    #   array([[ 1.04609058,  1.03806884], [ 1.03806884,  1.04609058]])
    slope = correlation * stdev_pred / stdev_obs
    intercept = mean_pred - slope * mean_obs

    # Mean Absolute Error (MAE)
    sum_abs = sum(abs(obs - pred))
    mae = sum_abs / n
    mse = sq_sum / n
    msea = intercept ** 2
    msei = 2 * intercept * (slope - 1) * mean_obs
    msep = (slope - 1) ** 2 * sum_obs / n
    mses = msea + msei + msep
    mseu = mse - mses

    # Root Mean Square Error (RMSE)
    # RMSES is a matrix like this:
    # array([[ 0.0154833 ,  0.01293596], [ 0.01293596,  0.0154833 ]])
    # RMSEU is a matrix like this:
    # array([[ 0.04147761,  0.0423413 ], [ 0.0423413 ,  0.04147761]])
    rmse = np.sqrt(mse)
    rmses = np.sqrt(mses)
    rmseu = np.sqrt(mseu)

    # Indices of agreement
    #   d1: Based on MAE
    #   d2: Based on RMSE
    pe1 = sum(abs(pred - mean_obs) + abs(obs - mean_obs))
    pe2 = sum((abs(pred - mean_obs) + abs(obs - mean_obs)) ** 2)
    d1 = 1 - n * mae / pe1
    d2 = 1 - n * mse / pe2

    # The output should be the following for this example:
    # [0.50413223140495866, 0.50067784684676719, 0.32746467365551984, 0.34255771110890154, 0.99233169655462861, -0.022646114907439507, 1.0380688421681805, 0.034876667833470279, 0.94108912881591422, 0.044273297270294019, 0.012935955298159947, 0.042341302668999582, 0.99561710541432158]
    y = [mean_obs, mean_pred, stdev_obs, stdev_pred,
         correlation.tolist()[0][1], intercept.tolist()[0][1], slope.tolist()[0][1],
         mae, d1, rmse, rmses.tolist()[0][1], rmseu.tolist()[0][1], d2]
    return y


# print("Mean Obs: ", evaluation[0])
# print("Mean Pred: ", evaluation[1])
# print("sdo: ", evaluation[2])
# print("sdp: ", evaluation[3])
# print("r: ", evaluation[4])
# print("a: ", evaluation[5])
# print("b: ", evaluation[6])
# print("mae: ", evaluation[7])
# print("d1: ", evaluation[8])
# print("rmse: ", evaluation[9])
# print("rmses: ", evaluation[10])
# print("rmseu: ", evaluation[11])
# print("d2: ", evaluation[12])