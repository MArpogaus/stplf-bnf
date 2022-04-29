# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : conftest.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-04-29 13:26:39 (Marcel Arpogaus)
# changed : 2021-03-26 11:48:25 (Marcel Arpogaus)
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

import itertools
import os

import numpy as np
import pandas as pd
import pytest

columns = ["ref", "load", "x1"]


def gen_value(column, line, id=0):
    if column == "ref":
        return id * 1e5 + line
    else:
        return np.random.randint(0, 1000)


def gen_df(columns, date_range, id=0, seed=1):
    np.random.seed(seed)
    periods = date_range.size
    df = pd.DataFrame(
        {
            "date_time": date_range,
            **{c: [gen_value(c, l, id) for l in range(periods)] for c in columns},
        }
    )
    return df


def gen_df_with_id(ids, columns, date_range):
    dfs = []
    for i in ids:
        df = gen_df(columns, date_range, i)
        df["id"] = i
        dfs.append(df)
    df = pd.concat(dfs)

    return df


@pytest.fixture(
    scope="function",
    params=[
        (list(range(5)), columns, 48 * 30),
    ],
)
def time_series_df_with_id(request):
    ids, columns, periods = request.param
    df = gen_df_with_id(
        ids=ids,
        columns=columns,
        date_range=pd.date_range("1/1/1", periods=periods, freq="30T"),
    )
    return df


@pytest.fixture
def artifical_cer_data(tmpdir_factory, time_series_df_with_id):
    tmpdir = tmpdir_factory.mktemp("csv_data")
    df = time_series_df_with_id

    start_date = df.date_time.dt.date.min()
    end_date = df.date_time.dt.date.max()
    date_range = pd.date_range(start_date, end_date, freq="D")

    df.set_index("date_time", inplace=True)

    months = len(date_range.to_period(freq="M").unique())
    split_date = end_date - int(np.round(0.1 * months)) * pd.offsets.MonthBegin()

    train_periods = date_range[date_range < split_date].size
    test_periods = date_range.size - train_periods

    test_data_split_day_str = str(split_date)
    train_data_split_day_str = str(split_date - pd.offsets.Minute(30))

    test_data = df.loc[test_data_split_day_str:]
    train_data = df.loc[:train_data_split_day_str]

    test_data.to_csv(tmpdir / "test.csv")
    train_data.to_csv(tmpdir / "train.csv")

    stats_path = tmpdir / "stats"
    os.mkdir(stats_path)
    stats = train_data.describe()
    stats.to_csv(stats_path / "train.csv")
    return tmpdir, train_periods, test_periods


@pytest.fixture(params=[0, 1, 48])
def history_size(request):
    return request.param


@pytest.fixture
def prediction_size(history_size):
    return history_size


@pytest.fixture
def shift(prediction_size):
    return prediction_size


@pytest.fixture(params=[32])
def batch_size(request):
    return request.param


@pytest.fixture(params=[[], ["ref"], ["load", "ref"], columns])
def history_columns(request):
    return request.param


@pytest.fixture(
    params=[
        [],
        list(
            itertools.chain(
                ["ref"],
                *[
                    [c + "_sin", c + "_cos"]
                    for c in ["weekday", "dayofyear", "time", "month"]
                ],
            )
        ),
    ]
)
def meta_columns(request):
    return request.param


@pytest.fixture
def prediction_columns(history_columns):
    return history_columns
