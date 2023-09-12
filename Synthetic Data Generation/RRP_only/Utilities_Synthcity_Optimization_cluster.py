# stdlib
import io
from pathlib import Path
from typing import List, Tuple

# third party
import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler

class EnergyDataloader:
    def __init__(self, seq_len: int=4, as_numpy: bool = False) -> None:
        self.seq_len = seq_len
        self.as_numpy = as_numpy

    def load(
        self, dataframe
    ) -> Tuple[pd.DataFrame, List[pd.DataFrame], List, pd.DataFrame]:
        # Load Energy Data
        #df = pd.read_csv(df_path)
        df = dataframe
        #df = df[:216]
        #df = df[::12].reset_index(drop=True)

        #df = df.drop(
            #columns=["REGION", "temperature", "RRP", "public_holidays", "precipitation", "relative_humidity", "soil_water_content",
                     #"solar_radiation", "wind_speed"])

        # Flip the data to make chronological data
        df = pd.DataFrame(df.values[::-1], columns=df.columns)


        T = (
            pd.to_datetime(df["SETTLEMENTDATE"], infer_datetime_format=True)
            .astype(np.int64)
            .astype(np.float64)
            / 10**9
        )

        #normalizing of the settlementdate
        T = pd.Series(MinMaxScaler().fit_transform(T.values.reshape(-1, 1)).squeeze())


        #normalizing of the temperature
        df = df.drop(columns=["SETTLEMENTDATE"])
        df = pd.DataFrame(MinMaxScaler().fit_transform(df), columns=df.columns)


        # Build dataset; dataX = list of temp & TOTALDEMAND, dataT = SETTLEMENTDATE
        dataX = []
        dataT = []
        outcome = []

        # Cut data by sequence length
        for i in range(0, len(df) - self.seq_len - 1):
            df_seq = df.loc[i : i + self.seq_len - 1]
            horizons = T.loc[i : i + self.seq_len - 1]
            out = df["TOTALDEMAND"].loc[i + self.seq_len]

            dataX.append(df_seq)
            dataT.append(horizons.values.tolist())
            outcome.append(out)

        # Mix Data (to make it similar to i.i.d)
        idx = np.random.permutation(len(dataX))

        temporal_data = []
        observation_times = []
        for i in range(len(dataX)):
            temporal_data.append(dataX[idx[i]])
            observation_times.append(dataT[idx[i]])


        if self.as_numpy:
            return (
                np.zeros((len(temporal_data), 0)),
                np.asarray(temporal_data, dtype=np.float32),
                np.asarray(observation_times),
                np.asarray(outcome, dtype=np.float32),
            )

        return (
            pd.DataFrame(np.zeros((len(temporal_data), 0))),
            temporal_data,
            observation_times,
            pd.DataFrame(outcome, columns=["TOTALDEMAND_next"]),
        )