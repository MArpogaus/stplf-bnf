# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : split.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-03-26 17:22:12 (Marcel Arpogaus)
# changed : 2021-04-22 18:33:44 (Marcel Arpogaus)
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

# Import libraries
import os
import sys

import yaml

import numpy as np

import pandas as pd

from tqdm import tqdm

# Load data ###################################################################
def load_data(data_path):
    data = pd.read_csv(
        data_path,
        parse_dates=["date_time"],
        infer_datetime_format=True,
        dtype={"id": "uint16", "load": "float32"},
    )
    return data


# Split in Training and Test data #############################################
def train_test_split(data, test_size):
    start_date = data.date_time.dt.date.min()
    end_date = data.date_time.dt.date.max()

    data.set_index("date_time", inplace=True)

    # Reserve some data for testing
    months = len(pd.period_range(start_date, end_date, freq="M"))
    split_date = end_date - int(np.round(test_size * months)) * pd.offsets.MonthBegin()
    tqdm.write(
        f"train data from {start_date} till {split_date - pd.offsets.Minute(30)}"
    )
    tqdm.write(f"test data from {split_date} till {end_date}")

    test_data_split_day_str = str(split_date)
    train_data_split_day_str = str(split_date - pd.offsets.Minute(30))
    test_data = data.loc[test_data_split_day_str:]
    train_data = data.loc[:train_data_split_day_str]

    return train_data, test_data


# save_data ###################################################################
def save_data(data, path):
    train_data, test_data = data
    it = tqdm(zip(["train", "test"], [train_data, test_data]), total=2)
    for ds, dat in it:
        file_path = os.path.join(path, f"{ds}.csv")
        dat.to_csv(file_path, float_format="%.3f")


# main ########################################################################
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(f"\tpython {sys.argv[0]} data-path\n")
        sys.exit(1)

    input = sys.argv[1]
    output = os.path.join("data", "split")

    params = yaml.safe_load(open("scripts/params.yaml"))["split"]

    if not os.path.exists(output):
        os.makedirs(output)

    steps = {
        "load data": {"fn": load_data},
        "train / test split": {
            "fn": train_test_split,
            "kwds": {"test_size": params["test_size"]},
        },
        "save data": {"fn": save_data, "kwds": {"path": output}},
    }

    it = tqdm(steps.items())
    for n, d in it:
        it.set_description(n)
        fn = d["fn"]
        output = fn(input, *d.get("args", []), **d.get("kwds", {}))
        input = output
