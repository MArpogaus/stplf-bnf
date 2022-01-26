# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : windowed_time_series_data_set.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-20 10:49:40 (Marcel Arpogaus)
# changed : 2022-01-20 15:47:03 (Marcel Arpogaus)
# DESCRIPTION #################################################################
# This file is part of the project "short-term probabilistic load
# forecasting using conditioned Bernstein-polynomial normalizing flows"
# LICENSE #####################################################################
# Short-Term Probabilistic Load Forecasting using Conditioned
# Bernstein-Polynomial Normalizing Flows (STPLF-BNF)
# Copyright (C) 2022 Marcel Arpogaus
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
###############################################################################

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf
import numpy as np

from ..loader import CSVDataLoader
from ..pipeline import WindowedTimeSeriesPipeline


# ref.: https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning
def encode(data, cycl_name, cycl_max, cycl=None):
    if cycl is None:
        cycl = getattr(data.index, cycl_name)
    data[cycl_name + "_sin"] = np.float32(np.sin(2 * np.pi * cycl / cycl_max))
    data[cycl_name + "_cos"] = np.float32(np.cos(2 * np.pi * cycl / cycl_max))
    return data


class DatasetGenerator:
    def __init__(self, df, columns):
        df.sort_index(inplace=True)
        self.grpd = df.groupby("id")
        self.columns = columns

    def __call__(self):
        for _, d in self.grpd:
            yield d[self.columns].values


class WindowedTimeSeriesDataSet:
    def __init__(
        self,
        history_size,
        prediction_size,
        history_columns,
        meta_columns,
        prediction_columns,
        file_path=None,
        data_splitter=None,
        column_transformers={},
        shift=None,
        batch_size=32,
        cycle_length=100,
        shuffle_buffer_size=1000,
        seed=42,
    ):
        self.columns = sorted(
            list(set(history_columns + prediction_columns + meta_columns))
        )

        if file_path:
            dtype = {
                "id": "uint16",
                "load": "float32",
                "is_holiday": "uint8",
                "weekday": "uint8",
            }

            if shift is None:
                shift = prediction_size
            else:
                shift = shift

            self.data_loader = CSVDataLoader(file_path=file_path, dtype=dtype)
            self.data_splitter = data_splitter

        else:
            self.data_loader = None
            self.data_splitter = None

        self.data_pipeline = WindowedTimeSeriesPipeline(
            history_size=history_size,
            prediction_size=prediction_size,
            history_columns=history_columns,
            meta_columns=meta_columns,
            prediction_columns=prediction_columns,
            shift=shift,
            batch_size=batch_size,
            cycle_length=cycle_length,
            shuffle_buffer_size=shuffle_buffer_size,
            seed=seed,
            column_transformers=column_transformers,
        )

    def __call__(self, data=None):
        if self.data_loader is not None and data is None:
            data = self.data_loader()
        elif data is not None:
            data = data
        else:
            ValueError("No data Provided")

        if "dayofyear_sin" in self.columns or "dayofyear_cos" in self.columns:
            data = encode(data, "dayofyear", 366)
        if "time_sin" in self.columns or "time_cos" in self.columns:
            data = encode(
                data,
                "time",
                23 * 60 + 59,
                cycl=data.index.hour * 60 + data.index.minute,
            )
        if ("weekday" in self.columns) and not "weekday" in data.columns:
            data["weekday"] = np.uint8(data.index.weekday)
        if "weekday_sin" in self.columns or "weekday_cos" in self.columns:
            data = encode(data, "weekday", 6.0, cycl=data.index.weekday)

        if self.data_splitter is not None:
            data = self.data_splitter(data)

        generator = DatasetGenerator(data, self.columns)
        ds = tf.data.Dataset.from_generator(
            generator,
            output_types=tf.float32,
            output_shapes=tf.TensorShape([None, len(self.columns)]),
        )
        ds = self.data_pipeline(ds)
        return ds
