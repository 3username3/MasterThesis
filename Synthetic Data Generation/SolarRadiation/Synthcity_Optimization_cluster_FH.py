import os
import pandas as pd
import datetime
import numpy as np
import torch
import optuna
from statistics import mean, median
import matplotlib.pyplot as plt

from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig, OutputType

import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
import rpy2.rinterface

from synthcity.metrics import Metrics
from Utilities_Synthcity_Optimization_cluster import EnergyDataloader
from synthcity.plugins.core.dataloader import TimeSeriesDataLoader

from optuna.visualization.matplotlib import plot_timeline

#import R packages
SlidingWindows = importr("SlidingWindows")
base = importr("base")

pd.set_option('display.max_columns', None)

# Recommended to train with a GPU
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)
#Needs to be set to true to prevent memory error - but impacts reproducibility. Necessary trade-off.
torch.backends.cudnn.benchmark = True


def data_preparation_energy_data(file_name, number_examples, max_sequence_length, number_features, use_xth_observation):
    data_df = pd.read_csv(file_name)
    date_df = data_df["SETTLEMENTDATE"]
    data_df = data_df.drop(columns=["REGION", "relative_humidity", "SETTLEMENTDATE", "precipitation", "public_holidays","RRP","soil_water_content","wind_speed"])
    data_np = data_df.to_numpy()
    data_float32 = np.float32(data_np).round(2)
    data_float32 = data_float32[::use_xth_observation]

    data_reshaped = data_float32.reshape(number_examples, max_sequence_length, number_features)

    print(f"shape energy data: {data_reshaped.shape} ")



    return date_df, data_float32, data_reshaped

def data_preparation_attributes(file_name_attributes):
    attribute_df = pd.read_csv(file_name_attributes)

    #drop axis and rows not needed
    attribute_df = attribute_df.drop("Date", axis=1).tail(-2).head(-1)
    attribute_np = attribute_df.to_numpy()

    print(f"shape attribute data: {attribute_np.shape} ")

    return attribute_df

def objective(trial):
    #index = int(os.environ['SLURM_ARRAY_TASK_ID'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = "cpu"
    print('Using device:', device)

    date_column, data_linear, data = data_preparation_energy_data("/home/ai21m017/datasets/train_energy_temp_data_NSW_values_per_day_24.csv", 13, 2190, 3, 4)
    #attributes = data_preparation_attributes("C:/dev/Masterarbeit/EnergyDataAUS/Energy_Attributes_NSW.csv")

    """
    DoppelGANger - train synthetic data 
    """

    model = DGAN(DGANConfig(
        max_sequence_len=2190,
        sample_len=trial.suggest_categorical("sample_len", [730, 365]),
        batch_size=trial.suggest_categorical("batch_size", [10, 50, 100]),
        epochs=trial.suggest_categorical("epochs", [100, 500, 1000]),
    ))


    # Train model
    model.train_numpy(features=data)

    print("Training done - generating starting.")

    # Generate values
    synthetic_values = model.generate_numpy(13)

    list_syn_values = []

    for first_index in range(13):
        for second_index in synthetic_values[1][first_index]:
                tuple = [*second_index, ]
                list_syn_values.append(tuple[0])

    syn_values_np_float32 = np.array(np.float32(list_syn_values))

    #get only date and demand column
    original_data = data_linear[:, :1]
    original_df = pd.DataFrame(original_data, columns=["TOTALDEMAND"])
    original_df = pd.concat([date_column, original_df], axis=1, join="inner")

    synthetic_data = pd.DataFrame(syn_values_np_float32, columns=["TOTALDEMAND"])
    synthetic_df = pd.concat([date_column, synthetic_data], axis=1, join="inner")

    """
     Statistical Checks
    """

    # calcualte mean
    mean_orig_train_data = mean(original_data[:, 0])
    mean_syn_data = mean(syn_values_np_float32)
    delta_mean_abs = abs(mean_orig_train_data - mean_syn_data)

    # calculate median
    median_orig_train_data = median(original_data[:, 0])
    median_syn_data = median(syn_values_np_float32)
    delta_median_abs = abs(median_orig_train_data - median_syn_data)

    """
    Detrended Cross Correlation Coefficient
    """
    def detrended_cross_correlation(original_data, synthetic_data):


        # reduce values per day to 2 - to decrease calculation time for rhodcca
        data_original = original_data[:, 0]
        #data_original = data_original[:320]
        syn_values = synthetic_data

        # import rpy2.robjects.numpy2ri - this casts numpy objects automatically to R objects
        rpy2.robjects.numpy2ri.activate()

        y1_vector = ro.FloatVector(data_original)
        y2_vector = ro.FloatVector(syn_values)

        rhodcca = SlidingWindows.rhodcca_SlidingWindows(y1_vector, y2_vector, w=292, nu=1)  # 292 is der kleinste m gliche Wert, da 250<w sein soll. Ist auch durch 4 teilbar
        print(rhodcca)
        #rhodcca = SlidingWindows.rhodcca_SlidingWindows(y1_vector, y2_vector, w=292, nu=1)  # w=292 is der kleinste m gliche Wert, da 250<w sein soll. Ist auch durch 4 teilbar
        # die 4 S ulen stehen f r die Position im Sliding Window, das nur n=4 gro  ist (?)

        column = [row[0] for row in rhodcca[2]]
        rhodcca_mean = mean(column)
        print(rhodcca_mean)
        return rhodcca_mean

    mean_rhodcca = detrended_cross_correlation(data_linear, syn_values_np_float32)

    """
    Synthcity - evaluating quality of synthetic data
    """

    Loader_Orig_EnergyData = EnergyDataloader()
    Loader_Syn_EnergyData = EnergyDataloader()

    #Synthcity Dataloader
    static_data_orig, temporal_data_orig, observation_times_orig, outcome_data_orig = Loader_Orig_EnergyData.load(original_df)
    static_data_syn, temporal_data_syn, observation_times_syn, outcome_data_syn = Loader_Syn_EnergyData.load(synthetic_df)

    original_loader = TimeSeriesDataLoader(
        temporal_data=temporal_data_orig,
        observation_times=observation_times_orig,
        static_data=static_data_orig,
        outcome=outcome_data_orig,
    )

    synthetic_loader = TimeSeriesDataLoader(
        temporal_data=temporal_data_syn,
        observation_times=observation_times_syn,
        static_data=static_data_syn,
        outcome=outcome_data_syn,
    )

    score = Metrics.evaluate(
        original_loader,
        synthetic_loader,
        reduction="mean",
        n_histogram_bins=10,
        metrics={
            "sanity": ["common_rows_proportion"],
        },
        task_type="time_series",
        random_state=0,
        workspace="/home/ai21m017/datasets/TrialSeries1/SolarRadiation/workspace",
        use_cache=True
    )

    scores_mean = score.iloc[:, 2]
    print(scores_mean)

    common_rows_proportion = score.iloc[0, 2]
    
    #welche Metriken sollen wir hier verwenden?
    print(scores_mean)

    # Write generated synthetic demand data to csv
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"optuna_syndata_trial_{len(study.trials)}_{now}.csv"
    synthetic_numpy_cv = np.savetxt(filename, syn_values_np_float32, delimiter=',', fmt='%f')
    
    text_file = open(f"optuna_parameter_trial_{len(study.trials)}_{now}.txt", "w")
    n = text_file.write(f"trial params: {trial.params}, delta_mean:{delta_mean_abs}, delta_median_abs:{delta_median_abs}, mean_rhodcca:{mean_rhodcca}, scores mean:{scores_mean} ")
    text_file.close()

    results =[mean_rhodcca, common_rows_proportion, delta_mean_abs, delta_median_abs]

    return results

print(f"objective:{objective}")

study = optuna.create_study(directions=["maximize", "minimize", "minimize","minimize"])
study.optimize(objective, n_trials=40, gc_after_trial=True)

print("Number of finished trials: {}".format(len(study.trials)))

trial = study.best_trials
print(f"Best trials: {trial}")

#plot

now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

plot_timeline(study)
plt.savefig(f"Timeline_{now}")