# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test_data.py
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
import os

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from bernstein_paper.data.cer_data_loader import load_data
from bernstein_paper.data.dataset import WindowedTimeSeriesDataSet
from bernstein_paper.data.dataset.windowed_time_series_data_set import encode
from bernstein_paper.data.splitter import TimeSeriesSplit

tf.random.set_seed(42)

columns = [
    "load",
    "time_sin",
    "time_cos",
    "weekday",
    "weekday_sin",
    "weekday_cos",
    "dayofyear_sin",
    "dayofyear_cos",
]

history_size = 48 * 7
prediction_size = 48
shift = prediction_size

history_columns = columns[:1]
meta_columns = columns[1:]
prediction_columns = columns[:1]

batch_size = 32
cycle_length = 1
shuffle_buffer_size = 0

ids = list(range(10))
periods = 48 * 30 * 6

date_range = pd.date_range("2001-01", periods=periods, freq="30min")
drf = date_range.to_frame()
drf_weekday = encode(drf, "weekday", 6, cycl=drf.index.weekday)
drf_dayofyear = encode(drf, "dayofyear", 366)
drf_time = encode(
    drf, "time", 23 * 60 + 59, cycl=drf.index.hour * 60 + drf.index.minute
)


def gen_value(id, column, line):
    if column == "weekday":
        return date_range[line].weekday()
    elif "weekday" in column:
        return drf_weekday.loc[date_range[line], column]
    elif "dayofyear" in column:
        return drf_dayofyear.loc[date_range[line], column]
    elif "time" in column:
        return drf_time.loc[date_range[line], column]
    else:
        col_num = columns.index(column)
        return int(f"{id:02d}{col_num:02d}{line:04d}")


load_max = gen_value(len(ids), "load", periods)


def infer_batch_from_first_value(start, patch_size):
    start = start.copy()
    batch = [start]
    for i in range(1, patch_size):
        batch.append(start + i)
    return np.stack(batch, 1)


def gen_batch_sorted(
    batch_no, bs, patch_size, patch_columns, extra_shift=0, periods=periods
):
    batch = []
    num_shifts = (periods - history_size) // shift
    num_patch = batch_no * batch_size
    for patch in range(bs):
        id = (num_patch + patch) // num_shifts
        val = shift * ((num_patch + patch) % num_shifts) + extra_shift
        b = [
            [gen_value(id, c, v) for c in sorted(patch_columns)]
            for v in range(val, val + patch_size)
        ]
        batch.append(b)
    return np.float32(batch)


def apply_trafo_to_columns(trafos, x, columns):
    x = x.copy()
    d = []
    for idx in range(x.shape[-1]):
        c = sorted(columns)[idx]
        if c in trafos.keys():
            d.append(trafos[c](x[..., idx, None]))
        else:
            d.append(x[..., idx, None])
    return np.concatenate(d, -1)


def apply_inv_trafo_to_columns(inv_trafos, x, columns):
    x = x.copy()
    for c, (s, it) in inv_trafos.items():
        if c in columns:
            idx = sorted(columns).index(c)
            # fmt: off
            x[..., idx:idx + s] = it(x[..., idx:idx + s])
            # fmt: on
    return x


def gen_df():
    dfs = []
    for i in ids:
        df = pd.DataFrame(
            {
                "date_time": date_range,
                "id": i,
                **{
                    c: [gen_value(i, c, l) for l in range(periods)]
                    for n, c in enumerate(columns)
                    if c != "weekday"
                },
            }
        )
        dfs.append(df)
    df = pd.concat(dfs)
    df["weekday"] = df.date_time.dt.weekday

    return df


@pytest.fixture(scope="session")
def test_data_simple(tmpdir_factory):
    file_path = tmpdir_factory.mktemp("csv_data") / "test.csv"
    df = gen_df()
    df.to_csv(file_path, index=False)

    return file_path


@pytest.fixture(scope="session")
def artifical_cer_data(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("csv_data")
    df = gen_df()
    start_date = df.date_time.dt.date.min()
    end_date = df.date_time.dt.date.max()

    df.set_index("date_time", inplace=True)

    months = len(pd.period_range(start_date, end_date, freq="M"))
    split_date = end_date - int(np.round(0.1 * months)) * pd.offsets.MonthBegin()

    train_periods = date_range[date_range < split_date].size
    test_periods = periods - train_periods

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


def test_dataset_artificial_data(test_data_simple):

    data_set = WindowedTimeSeriesDataSet(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
        file_path=test_data_simple,
        shift=shift,
        batch_size=batch_size,
        cycle_length=cycle_length,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=42,
    )
    ds = data_set()
    (x1, x2), y = next(ds.take(1).as_numpy_iterator())
    assert x1.shape == (
        batch_size,
        history_size,
        len(history_columns),
    ), "Wrong shape: historic data"
    assert x2.shape == (
        batch_size,
        1,
        len(meta_columns),
    ), "Wrong shape: meta data"
    assert y.shape == (
        batch_size,
        prediction_size,
        len(prediction_columns),
    ), "Wrong shape: meta data"

    for b, ((x1, x2), y) in enumerate(ds.as_numpy_iterator()):
        bs = x1.shape[0]
        x1_test = infer_batch_from_first_value(x1[:, 0, :], history_size)
        assert np.allclose(x1, x1_test), f"Wrong data: history ({b})"
        assert np.allclose(
            x1, gen_batch_sorted(b, bs, history_size, history_columns)
        ), f"Wrong data: history ({b})"
        assert np.allclose(
            x2, gen_batch_sorted(b, bs, 1, meta_columns, history_size + 1)
        ), f"Wrong data: meta ({b})"
        assert np.allclose(
            y, x1[:, -prediction_size:] + prediction_size
        ), f"Wrong data: prediction ({b})"
        assert np.allclose(
            y,
            gen_batch_sorted(b, bs, prediction_size, prediction_columns, history_size),
        ), f"Wrong data: prediction ({b})"


def test_dataset_artificial_data_column_transformer(test_data_simple):

    column_transformers = {
        "load": lambda x: x / load_max,
        "weekday": lambda x: tf.one_hot(tf.cast(x[..., 0], tf.uint8), 7),
    }
    inv_column_transformers = {
        "load": (1, lambda x: x * load_max),
        "weekday": (7, lambda x: tf.math.argmax(x)),
    }
    data_splitter = TimeSeriesSplit(0.5, TimeSeriesSplit.LEFT)

    data_set = WindowedTimeSeriesDataSet(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
        file_path=test_data_simple,
        data_splitter=data_splitter,
        column_transformers=column_transformers,
        shift=shift,
        batch_size=batch_size,
        cycle_length=cycle_length,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=42,
    )
    ds = data_set()
    (x1, x2), y = next(ds.take(1).as_numpy_iterator())
    assert x1.shape == (
        batch_size,
        history_size,
        len(history_columns),
    ), "Wrong shape: historic data"
    assert x2.shape == (
        batch_size,
        1,
        len(meta_columns) + 6,
    ), "Wrong shape: meta data"
    assert y.shape == (
        batch_size,
        prediction_size,
        len(prediction_columns),
    ), "Wrong shape: meta data"

    for b, ((x1, x2), y) in enumerate(ds.as_numpy_iterator()):
        bs = x1.shape[0]
        x1_test = apply_trafo_to_columns(
            column_transformers,
            infer_batch_from_first_value(
                apply_inv_trafo_to_columns(
                    inv_column_transformers, x1[:, 0, :], history_columns
                ),
                history_size,
            ),
            history_columns,
        )
        assert np.allclose(x1, x1_test), f"Wrong data: history ({b})"
        x1_test = apply_trafo_to_columns(
            column_transformers,
            gen_batch_sorted(
                b, bs, history_size, history_columns, periods=periods // 2
            ),
            history_columns,
        )
        assert np.allclose(x1, x1_test), f"Wrong data: history ({b})"
        x2_test = apply_trafo_to_columns(
            column_transformers,
            gen_batch_sorted(
                b, bs, 1, meta_columns, history_size + 1, periods=periods // 2
            ),
            meta_columns,
        )
        assert np.allclose(x2, x2_test), f"Wrong data: meta ({b})"

        assert np.allclose(
            y, x1[:, -prediction_size:] + prediction_size / load_max
        ), f"Wrong data: prediction ({b})"

        y_test = apply_trafo_to_columns(
            column_transformers,
            gen_batch_sorted(
                b,
                bs,
                prediction_size,
                prediction_columns,
                history_size,
                periods=periods // 2,
            ),
            prediction_columns,
        )
        assert np.allclose(y, y_test), f"Wrong data: prediction ({b})"


def test_artificial_data(artifical_cer_data):
    tmpdir, train_periods, test_periods = artifical_cer_data

    split_day = pd.Timestamp(date_range[int(np.ceil(train_periods * 0.9))].date())
    split_periods = {
        "train": (date_range < split_day).sum(),
        "validate": train_periods - (date_range < split_day).sum(),
        "test": test_periods,
    }
    meta_columns = ["dayofyear_sin", "dayofyear_cos", "weekday"]

    data_stats = pd.read_csv(tmpdir / "stats" / "train.csv", index_col=0)
    load_max = data_stats.loc["max", "load"]
    column_transformers = {
        "load": lambda x: x / load_max,
        "weekday": lambda x: x / 6,
    }
    inv_column_transformers = {
        "load": (1, lambda x: x * load_max),
        "weekday": (1, lambda x: x * 6),
    }
    data = load_data(
        data_path=tmpdir,
        data_stats_path=tmpdir / "stats",
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
        shift=shift,
        validation_split=0.1,
        batch_size=batch_size,
        cycle_length=cycle_length,
        shuffle_buffer_size=shuffle_buffer_size,
        seed=42,
    )

    assert ["train", "validate", "test"] == list(data.keys()), "Unexpected Splits"

    for split, ds in data.items():
        (x1, x2), y = next(ds.take(1).as_numpy_iterator())
        assert x1.shape == (
            batch_size,
            history_size,
            len(history_columns),
        ), f"Wrong shape: historic data ({split})"
        assert x2.shape == (
            batch_size,
            1,
            len(meta_columns),
        ), f"Wrong shape: meta data ({split})"
        assert y.shape == (
            batch_size,
            prediction_size,
            len(prediction_columns),
        ), f"Wrong shape: meta data ({split})"

        for b, ((x1, x2), y) in enumerate(ds.as_numpy_iterator()):
            extra_shift = 0
            if split == "validate":
                extra_shift = split_periods["train"]
            elif split == "test":
                extra_shift = train_periods
            bs = x1.shape[0]
            x1_test = apply_trafo_to_columns(
                column_transformers,
                infer_batch_from_first_value(
                    apply_inv_trafo_to_columns(
                        inv_column_transformers, x1[:, 0, :], history_columns
                    ),
                    history_size,
                ),
                history_columns,
            )
            assert np.allclose(x1, x1_test), f"Wrong data: history ({b}, {split})"
            x1_test = apply_trafo_to_columns(
                column_transformers,
                gen_batch_sorted(
                    b,
                    bs,
                    history_size,
                    history_columns,
                    extra_shift=extra_shift,
                    periods=split_periods[split],
                ),
                history_columns,
            )
            assert np.allclose(x1, x1_test), f"Wrong data: history ({b}, {split})"
            x2_test = apply_trafo_to_columns(
                column_transformers,
                gen_batch_sorted(
                    b,
                    bs,
                    1,
                    meta_columns,
                    extra_shift=extra_shift + history_size + 1,
                    periods=split_periods[split],
                ),
                meta_columns,
            )
            assert np.allclose(x2, x2_test), f"Wrong data: meta ({b}, {split})"

            assert np.allclose(
                y, x1[:, -prediction_size:] + prediction_size / load_max
            ), f"Wrong data: prediction ({b}, {split})"

            y_test = apply_trafo_to_columns(
                column_transformers,
                gen_batch_sorted(
                    b,
                    bs,
                    prediction_size,
                    prediction_columns,
                    extra_shift=extra_shift + history_size,
                    periods=split_periods[split],
                ),
                prediction_columns,
            )
            assert np.allclose(y, y_test), f"Wrong data: prediction ({b}, {split})"
            if split != "test":
                assert np.all(
                    (x1 >= 0) & (x1 <= 1)
                ), f"Wrong scaling: history ({b}, {split})"
