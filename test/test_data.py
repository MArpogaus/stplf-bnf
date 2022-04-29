# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test_data.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-20 10:49:40 (Marcel Arpogaus)
# changed : 2022-03-12 11:27:29 (Marcel Arpogaus)
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
from tensorflow_time_series_dataset.preprocessors import (
    CyclicalFeatureEncoder, GroupbyDatasetGenerator, TimeSeriesSplit)
from tensorflow_time_series_dataset.utils.test import (get_ctxmgr,
                                                       validate_dataset)

from bernstein_paper.data.cer_data_loader import load_data


def test_artificial_data(
    artifical_cer_data,
    batch_size,
    history_size,
    prediction_size,
    shift,
    history_columns,
    meta_columns,
    prediction_columns,
):
    cer_data_path, train_periods, test_periods = artifical_cer_data

    data_stats = pd.read_csv(cer_data_path / "stats" / "train.csv", index_col=0)
    load_max = data_stats.loc["max", "load"]
    column_transformers = {
        "load": lambda x: x / load_max,
    }

    cer_kwds = dict(
        data_path=cer_data_path,
        data_stats_path=cer_data_path / "stats",
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
        shift=shift,
        validation_split=0.1,
        batch_size=batch_size,
        cycle_length=5,
        shuffle_buffer_size=1000,
        seed=42,
    )

    encs = {
        "weekday": dict(cycl_max=6),
        "dayofyear": dict(cycl_max=366, cycl_min=1),
        "month": dict(cycl_max=12, cycl_min=1),
        "time": dict(
            cycl_max=24 * 60 - 1,
            cycl_getter=lambda df, k: df.index.hour * 60 + df.index.minute,
        ),
    }
    with get_ctxmgr(
        history_size=history_size,
        prediction_size=prediction_size,
        history_columns=history_columns,
        meta_columns=meta_columns,
        prediction_columns=prediction_columns,
    ):
        data = load_data(**cer_kwds)
        assert ["train", "validate", "test"] == list(data.keys()), "Unexpected Splits"

        for split, cer_ds in data.items():
            print(f"testing {split} data", end="...")
            if split in ("train", "validate"):
                cer_data_file = os.path.join(cer_data_path, "train.csv")
            else:
                cer_data_file = os.path.join(cer_data_path, "test.csv")
            df = pd.read_csv(
                cer_data_file,
                index_col="date_time",
                parse_dates=["date_time"],
                infer_datetime_format=True,
            )
            if cer_kwds.get("validation_split", False) and split in (
                "train",
                "validate",
            ):
                print("has validation_split", end="...")
                split_size = 1 - cer_kwds["validation_split"]
                splitter = TimeSeriesSplit(
                    split_size,
                    TimeSeriesSplit.LEFT if split == "train" else TimeSeriesSplit.RIGHT,
                )
                df = splitter(df)
            for c, f in column_transformers.items():
                df[c] = df[c].apply(f)
            for name, kwds in encs.items():
                enc = CyclicalFeatureEncoder(name, **kwds)
                df = enc(df)

            b = validate_dataset(
                df.sample(frac=1),
                cer_ds,
                **cer_kwds,
            )

            periods = df.index.unique().size
            window_size = history_size + prediction_size
            initial_size = window_size - shift
            patch_data_per_id = periods - initial_size
            patches = patch_data_per_id / shift * df.id.unique().size
            batches = int(patches // cer_kwds["batch_size"])
            assert b == batches, f"Note enough batches ({split})"
            print("OK")
