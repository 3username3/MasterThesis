import re
import pandas as pd
import numpy as np
from sklearn.metrics import median_absolute_error, mean_absolute_error
from SeqMetrics import RegressionMetrics
from darts.metrics import mape
import matplotlib.pyplot as plt
import datetime
pd.set_option('display.max_columns', None)


pd.set_option('display.max_columns', None)


def read_in_data(file_orig_data, xth_observation_orig):
    #prepare original data
    original_data = pd.read_csv(file_orig_data)
    original_data = original_data[:26280]

    data_points_per_day = int(24/xth_observation_orig)


    original_data = original_data[["SETTLEMENTDATE", "TOTALDEMAND", "public_holidays"]]
    original_data["SETTLEMENTDATE"] = pd.to_datetime(original_data["SETTLEMENTDATE"])
    original_data["Weekday"] = original_data["SETTLEMENTDATE"].dt.dayofweek
    original_data["Hour_of_day"] = original_data["SETTLEMENTDATE"].dt.hour
    original_data["Day_of_year"] = original_data["SETTLEMENTDATE"].dt.dayofyear


    original_df = original_data[::xth_observation_orig].reset_index(drop=True)
    date_df = original_df[["SETTLEMENTDATE", "public_holidays"]]

    print(original_df)

    #noriginal_df = original_df.set_index("SETTLEMENTDATE")

    # prepare synthetic data

    #synthetic_data = pd.read_csv(file_syn_data, names=["TOTALDEMAND"])
    #synthetic_df = pd.concat([date_df, synthetic_data], axis=1, join="inner")

    return original_df, date_df, data_points_per_day

def read_in_synthetic_data(file_syn_data, date_df, xth_observation_orig):
    #prepare original data
    syn_data = pd.read_csv(file_syn_data, names=["TOTALDEMAND"])
    syn_data = syn_data[::6570]

    syn_data = syn_data[::xth_observation_orig].reset_index(drop=True)

    syn_data["SETTLEMENTDATE"] = date_df["SETTLEMENTDATE"]
    syn_data["public_holidays"] = date_df["public_holidays"]
    syn_data["SETTLEMENTDATE"] = pd.to_datetime(syn_data["SETTLEMENTDATE"])

    syn_data = syn_data.reindex(columns=["SETTLEMENTDATE", "TOTALDEMAND", "public_holidays"])

    syn_data["Weekday"] = syn_data["SETTLEMENTDATE"].dt.dayofweek
    syn_data["Hour_of_day"] = syn_data["SETTLEMENTDATE"].dt.hour
    syn_data["Day_of_year"] = syn_data["SETTLEMENTDATE"].dt.dayofyear

    print(syn_data)

    return syn_data

def mase(original_data, forecast):

    original_data = np.float32(original_data)
    forecast = np.float32(forecast)

    #Create datasets for calculation of naive prediction
    naive_y = original_data[0:-2]
    naive_prediction = original_data[1:-1]

    # Calculate MAE (in sample), MAE and MASE
    mae_in_sample = np.mean(np.abs((naive_y - naive_prediction)))
    mae = np.mean(np.abs((original_data - forecast)))
    mase = np.divide(mae, mae_in_sample)

    return mase

def evaluation(load, model, future_covariates, data_points):


    backtest = model.historical_forecasts(series=load,
                                          future_covariates=future_covariates,
                                          retrain=True,
                                          verbose=True,
                                          forecast_horizon=data_points)

    load_series = load.pd_series()
    backtest_series = backtest.pd_series()

    # calculate forecast metrics
    er = RegressionMetrics(load_series[-len(backtest_series):], backtest_series)
    gmae_backtest = er.gmae()
    mdae_backtest = median_absolute_error(load_series[-len(backtest_series):], backtest_series)
    mae_backtest = mean_absolute_error(load_series[-len(backtest_series):], backtest_series)
    sMAPE_backtest = er.smape()
    RMSSE_backtest = er.rmsse()
    RMAE_backtest = (mae_backtest / 498.628)  # 498.28 is the MAE of the best naive seasonal forecastin model

    mape_backtest = mape(load, backtest)
    mase_backtest = mase(load_series[-len(backtest_series):], backtest_series)

    # error = abs(load_series[-len(backtest_series):] - backtest_series)
    # rolling_error = error.rolling(12, center=True).mean()

    list_metrics = [gmae_backtest, mdae_backtest, mae_backtest, sMAPE_backtest, RMSSE_backtest, RMAE_backtest,
                    mase_backtest, mape_backtest]

    print("Backtest GMAE = {}".format(gmae_backtest))
    print("Backtest MDAE = {}".format(mdae_backtest))
    print("Backtest MAE = {}".format(mae_backtest))

    print("Backtest sMAPE = {}".format(sMAPE_backtest))
    print("Backtest RMSSE = {}".format(RMSSE_backtest))
    print("Backtest RMAE = {}".format(RMAE_backtest))

    print("Backtest MASE = {}".format(mase_backtest))
    print("Backtest MAPE = {}".format(mape_backtest))

    return backtest_series, list_metrics, load_series

def save_as_csv(model_name, data, data_name):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    #filename = f"{model_name}_{data_name}_{now}.csv"
    filename = f"{model_name}_{data_name}_{now}.csv"
    np.savetxt(filename, data, delimiter=',', fmt='%s')

def split_filename(filepath):
    file_name_split_1 = re.split("/", filepath)
    file_name_split_2 = re.split("\.", file_name_split_1[-1])
    file_name = f"{file_name_split_1[-2]}_{file_name_split_2[0]}"
    return file_name