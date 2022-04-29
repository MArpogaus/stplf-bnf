# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : cer_data_loader.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-07-29 17:57:39 (Marcel Arpogaus)
# changed : 2022-03-12 11:10:57 (Marcel Arpogaus)
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
from tensorflow_time_series_dataset import WindowedTimeSeriesDatasetFactory
from tensorflow_time_series_dataset.loaders import CSVDataLoader
from tensorflow_time_series_dataset.preprocessors import (
    CyclicalFeatureEncoder, GroupbyDatasetGenerator, TimeSeriesSplit)

enc_kwds = {
    "weekday": dict(cycl_max=6),
    "dayofyear": dict(cycl_max=366, cycl_min=1),
    "month": dict(cycl_max=12, cycl_min=1),
    "time": dict(
        cycl_max=24 * 60 - 1,
        cycl_getter=lambda df, k: df.index.hour * 60 + df.index.minute,
    ),
}


def get_factory(data_stats_path: str, meta_columns, scale_load, **kwds):
    factory = WindowedTimeSeriesDatasetFactory(meta_columns=meta_columns, **kwds)

    for name, kwds in enc_kwds.items():
        if name + "_sin" in meta_columns and name + "_cos" in meta_columns:
            factory.add_preprocessor(CyclicalFeatureEncoder(name, **kwds))

    if scale_load:
        data_stats_file = os.path.join(data_stats_path, "train.csv")
        data_stats = pd.read_csv(data_stats_file, index_col=0)
        load_max = data_stats.loc["max", "load"]

        def load_scaler(data):
            data = data.copy()
            data.load = data.load.apply(lambda x: x / load_max)
            return data

        factory.add_preprocessor(load_scaler)
    return factory


def load_data(
    data_path: str,
    data_stats_path: str,
    history_size,
    prediction_size,
    history_columns=["load", "is_holiday", "tempC"],
    meta_columns=["is_holiday"],
    prediction_columns=["load"],
    splits=["train", "validate", "test"],
    shift=None,
    validation_split=None,
    batch_size=32,
    cycle_length=10,
    shuffle_buffer_size=1000,
    seed=42,
    test_mode=False,
    scale_load=True,
):
    """
    Loads the preprocessed CER data and build the dataset.

    :param      data_path:            The path to the folder containing the
                                      train.csv and test.csv
    :type       data_path:            str
    :param      history_size:         The number of time steps of the historic
                                      data a patch should contain
    :type       history_size:         int
    :param      prediction_size:      The number of time steps in the
                                      prediction horizon a step should contain
    :type       prediction_size:      int
    :param      history_columns:     The historic columns
    :type       history_columns:     Array
    :param      meta_columns:         The column names to be used as horizon
                                      data.
    :type       meta_columns:         Array
    :param      prediction_columns:   The columns to predict
    :type       prediction_columns:   Array
    :param      splits:               The data splits to be generated. At least
                                      one of 'train', 'validate' or 'test'
    :type       splits:               Array
    :param      shift:                The amount of time steps by which the
                                      window moves on each iteration
    :type       shift:                int
    :param      validation_split:     The amount of data reserved from the
                                      training set for validation
    :type       validation_split:     float
    :param      batch_size:           The batch size
    :type       batch_size:           int
    :param      cycle_length:         The number of input elements that are
                                      processed concurrently
    :type       cycle_length:         int
    :param      shuffle_buffer_size:  The shuffle buffer size
    :type       shuffle_buffer_size:  int
    :param      seed:                 The seed used by the pseudo random
                                      generators
    :type       seed:                 int

    :returns:   A dict containing the windowed TensorFlow datasets generated
                from csv file in `data_path` for the given `spits`.
    :rtype:     dict
    """
    # common ##################################################################
    data = {}
    factory_kwds = dict(
        data_stats_path=data_stats_path,
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
        scale_load=scale_load,
    )
    gdg = GroupbyDatasetGenerator(
        "id", columns=history_columns + meta_columns + prediction_columns
    )

    # train data ##############################################################
    train_data_path = os.path.join(data_path, "train.csv")
    test_data_path = os.path.join(data_path, "test.csv")

    if "train" in splits:
        train_factory = get_factory(**factory_kwds)
        if validation_split is not None:
            train_factory.add_preprocessor(
                TimeSeriesSplit(1 - validation_split, TimeSeriesSplit.LEFT)
            )
        train_factory.add_preprocessor(gdg)
        train_factory.set_data_loader(CSVDataLoader(file_path=train_data_path))
        data["train"] = train_factory.get_dataset()

    # validation data #########################################################
    if "validate" in splits and validation_split is not None:
        validation_factory = get_factory(**factory_kwds)
        if validation_split is not None:
            validation_factory.add_preprocessor(
                TimeSeriesSplit(1 - validation_split, TimeSeriesSplit.RIGHT)
            )
        validation_factory.add_preprocessor(gdg)
        validation_factory.set_data_loader(CSVDataLoader(file_path=train_data_path))
        data["validate"] = validation_factory.get_dataset()

    # test data ###############################################################
    if "test" in splits:
        test_factory = get_factory(**factory_kwds)
        test_factory.add_preprocessor(gdg)
        test_factory.set_data_loader(CSVDataLoader(file_path=test_data_path))
        data["test"] = test_factory.get_dataset()

    if len(data.keys()) == 0:
        raise ValueError("Invalid splits")

    return data
