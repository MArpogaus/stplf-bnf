# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test_baseline.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-04-29 13:25:33 (Marcel Arpogaus)
# changed : 2022-04-29 13:29:01 (Marcel Arpogaus)
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

from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from conftest import columns
from tensorflow_time_series_dataset.preprocessors import CyclicalFeatureEncoder

from bernstein_paper.data.cer_data_loader import enc_kwds
from bernstein_paper.models.baseline import build_model
from bernstein_paper.util import get_time_step


@pytest.fixture(params=[None, 0.1])
def validation_split(request):
    return request.param


@pytest.fixture(params=[True, False])
def scale_load(request):
    return request.param


def get_ctxmgr(prediction_columns, meta_columns, train_periods, validation_split):
    if validation_split is None:
        validation_split = 0
    days = np.floor(train_periods * (1 - validation_split))
    if len(prediction_columns) != 1:
        ctxmgr = pytest.raises(
            AssertionError,
            match="Only one prediction variable supported",
        )
    elif not (("dayofyear_sin" in meta_columns) and ("dayofyear_cos" in meta_columns)):
        ctxmgr = pytest.raises(
            AssertionError,
            match="dayofyear as meta column required",
        )
    elif days < 365:
        ctxmgr = pytest.raises(
            AssertionError,
            match="whole year of reference data required",
        )
    else:
        ctxmgr = does_not_raise()
    return ctxmgr


def gen_df(columns, date_range, id=0):
    lines = id * 1e6 + date_range.dayofyear * 100 + get_time_step(date_range)
    df = pd.DataFrame(
        {
            "date_time": date_range,
            **{c: [l for l in lines] for c in columns},
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
        (list(range(5)), columns, 48 * 450),
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


def refference_data(doy, num_ids):
    test = (
        np.arange(48)[None, ..., None] * np.ones((doy.size, 48, num_ids))
        + doy[..., None, None] * 100
    )
    return test


def test_baseline(
    artifical_cer_data,
    meta_columns,
    prediction_columns,
    validation_split,
    scale_load,
    batch_size,
):
    cer_data_path, train_periods, _ = artifical_cer_data

    with get_ctxmgr(prediction_columns, meta_columns, train_periods, validation_split):
        bm = build_model(
            data_path=cer_data_path,
            data_stats_path=cer_data_path / "stats",
            prediction_columns=prediction_columns,
            meta_columns=meta_columns,
            validation_split=validation_split,
            scale_load=scale_load,
        )

        enc = CyclicalFeatureEncoder("dayofyear", **enc_kwds["dayofyear"])
        x_test = np.random.normal(0, 1, (batch_size, len(meta_columns) - 2))
        doy = np.random.randint(1, 366, batch_size)

        test_df = pd.DataFrame(
            x_test, columns=list(filter(lambda x: "dayofyear" not in x, meta_columns))
        )
        test_df["dayofyear"] = doy
        test_df = enc(test_df)[sorted(meta_columns)]

        y = bm((None, test_df.values))
        assert y.shape[:-1] == [batch_size, 48], "wrong shape"

        ids = y[0, 0] // 1e6
        assert np.all((y // 1e6) == ids), "wrong ids"

        y_test = refference_data(doy, ids.shape[0])
        y *= bm.load_max
        assert np.allclose(y % 1e6, y_test), "wrong data"
