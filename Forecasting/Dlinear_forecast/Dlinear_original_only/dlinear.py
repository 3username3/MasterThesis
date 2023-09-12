from Utilities_Forecasting_Dlinear import *
import optuna
from darts import TimeSeries

from darts.models import (
    DLinearModel)

from darts.dataprocessing.transformers import Scaler
from darts.utils.model_selection import train_test_split

if __name__ == "__main__":

    original_df, date_df, measurements_per_day = read_in_data(
        "/home/ai21m017/datasets/Forecasting/train_energy_temp_data_NSW_values_per_day_24.csv",
        8)

    train_set, val_set = train_test_split(data=original_df, test_size=0.2)

    load_train = TimeSeries.from_dataframe(df=train_set, value_cols=["TOTALDEMAND"])
    load_val = TimeSeries.from_dataframe(df=val_set, value_cols=["TOTALDEMAND"])

    future_covs_train = TimeSeries.from_dataframe(df=train_set, value_cols=["Weekday", "Day_of_year", "Hour_of_day",
                                                                            "public_holidays"])
    future_covs_val = TimeSeries.from_dataframe(df=val_set, value_cols=["Weekday", "Day_of_year", "Hour_of_day",
                                                                        "public_holidays"])


    """
     Dlinear Model
     """
    results_list = []
    def objective(trial):
        dlinear_model = DLinearModel(
            #original values
            #input_chunk_length=trial.suggest_categorical("input_length", [(365*measurements_per_day), (30*measurements_per_day)]),

            input_chunk_length=trial.suggest_categorical("input_length", [(7 * measurements_per_day), (30 * measurements_per_day)]),

            output_chunk_length=trial.suggest_categorical("output_length", [measurements_per_day, (2*measurements_per_day)]),
            shared_weights=False,
            kernel_size=25,
            n_epochs=trial.suggest_categorical("epochs", [500, 1000]),
            const_init=True,
            random_state=42,
            use_static_covariates=True,
            pl_trainer_kwargs={"accelerator": "gpu", "devices": -1, "auto_select_gpus": True}
        )

        dlinear_model.fit(series=load_train, future_covariates=future_covs_train)

        trial_number = len(study.trials)
        model_name = f"dlinear_model_trial{trial_number}"

        backtest_naive, backtest_metrics, load_series = evaluation(load_val, dlinear_model, date_df, future_covs_val, model_name, measurements_per_day)

        now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

        params_results = backtest_metrics.copy()
        params_results.append(trial.params)
        params_results.append(f"trial_{trial_number}_{now}")
        results_list.append(params_results)

        #MAPE is calculated too, but is not used as optimization parameter
        results = params_results[0:4]


        save_as_csv(model_name, params_results, f"params_results")
        save_as_csv(model_name, load_series, f"load")
        save_as_csv(model_name, backtest_naive, f"backtest")

        return results

    study = optuna.create_study(directions=["minimize", "minimize", "minimize", "minimize"])

    study.optimize(objective, n_trials=50, gc_after_trial=True)

    save_as_csv("dlinear_model", results_list, "results")

    print("Number of finished trials: {}".format(len(study.trials)))

    trial = study.best_trials
    model_name = "dlinear"

    print(f"Best trials for {model_name}: {trial}")