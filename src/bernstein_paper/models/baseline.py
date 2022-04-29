# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : baseline.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-04-29 13:23:45 (Marcel Arpogaus)
# changed : 2022-04-29 13:28:50 (Marcel Arpogaus)
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
import os

import pandas as pd
import tensorflow as tf
from tensorflow_time_series_dataset.preprocessors import (
    CyclicalFeatureEncoder, TimeSeriesSplit)

from bernstein_paper.data.cer_data_loader import enc_kwds


def get_time_step(dt):
    return (dt.hour * 60 + dt.minute) // 30


class BaselineModel(tf.keras.Model):
    def __init__(
        self, train_data, prediction_column, meta_columns, load_max=1, name=None
    ):
        super().__init__(name=name)

        assert ("dayofyear_sin" in meta_columns) and (
            "dayofyear_cos" in meta_columns
        ), "dayofyear as meta column required"

        ts = get_time_step(train_data.index)
        doy = train_data.index.day_of_year

        assert doy.unique().size == 365, "whole year of reference data required"

        self._lut = train_data.groupby([doy, ts, "id"])[prediction_column].agg("first")
        self._lut = self._lut.sort_index().values.reshape(
            (doy.unique().size, ts.unique().size, -1)
        )
        self._lut = tf.convert_to_tensor(self._lut / load_max)
        self.load_max = load_max

        self._meta_columns_idx = {c: i for i, c in enumerate(sorted(meta_columns))}
        self._enc = CyclicalFeatureEncoder("", **enc_kwds["dayofyear"])

    def get_samples(self, doy):
        doy = tf.cast(doy, tf.int32) - 1
        samples = tf.gather(self._lut, doy, axis=0)
        return samples

    def call(self, x):
        _, x2 = x
        doy_sin = tf.squeeze(x2)[..., self._meta_columns_idx["dayofyear_sin"]]
        doy_cos = tf.squeeze(x2)[..., self._meta_columns_idx["dayofyear_cos"]]

        doy = tf.round(self._enc.decode(doy_sin, doy_cos))

        return self.get_samples(doy)


def build_model(
    data_path: str,
    data_stats_path: str,
    prediction_columns,
    meta_columns,
    validation_split=None,
    scale_load=True,
    name=None,
    **unused_kwds
):

    csv_path = os.path.join(data_path, "train.csv")
    train_data = pd.read_csv(
        csv_path,
        parse_dates=["date_time"],
        infer_datetime_format=True,
        index_col=["date_time"],
        dtype={
            "id": "uint16",
            "load": "float32",
            "is_holiday": "uint8",
            "weekday": "uint8",
        },
    )
    if scale_load:
        data_stats = pd.read_csv(
            os.path.join(data_stats_path, "train.csv"), index_col=0
        )
        load_max = data_stats.loc["max", "load"]
    else:
        load_max = 1
    if validation_split:
        splitter = TimeSeriesSplit(1 - validation_split, TimeSeriesSplit.LEFT)
        train_data = splitter(train_data)

    assert len(prediction_columns) == 1, "Only one prediction variable supported"

    return BaselineModel(
        train_data=train_data,
        prediction_column=prediction_columns[0],
        meta_columns=meta_columns,
        load_max=load_max,
        name=name,
    )
