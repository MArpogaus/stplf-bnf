# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : validate_data.py
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

# Import libraries
import os
import sys

import pandas as pd
from tqdm import tqdm


# Load data ###################################################################
def load_data(data_path):
    data = pd.read_csv(
        data_path,
        parse_dates=["date_time"],
        infer_datetime_format=True,
        dtype={
            "id": "uint16",
            "load": "float32",
            "tempC": "int8",
            "is_holiday": "uint8",
        },
    )
    return data


def test_invalid_data(data):
    assert not data.isnull().any().any(), "Data contains NaNs"
    assert not data.load[data.load < 0.0].any(), "Data contains negative loads values"
    return data


def test_missing_values(data):
    assert (
        data.groupby("id").size().unique().size == 1
    ), "some records are missing values"
    assert data.groupby(["id", data.date_time.dt.date]).size().unique() == [
        48
    ], "Some days are missing values"
    return data


def test_information_leakage(train_data, test_data):

    assert test_data.index.isin(
        train_data.index.unique()
    ).any(), "Information leak detected!"

    # ### Are ids in both sets?

    assert (
        test_data.id.unique() in train_data.id.unique()
        and train_data.id.unique() in test_data.id.unique()
    ), "not all assets in both sets"


# save stats ##################################################################
def save_stats(data, name, path):
    file_path = os.path.join(path, name + ".csv")
    stats = data.reset_index().describe()
    stats.loc["unique.size"] = data.agg(lambda x: x.unique().size)
    stats.to_csv(file_path, float_format="%.3f")
    return data


# save max load ###############################################################
def save_max_load(data, name, path):
    file_path = os.path.join(path, f"{name}_max_load")
    with open(file_path, "w+") as f:
        f.write(str(data.load.max()))

    return data


# main ########################################################################
if __name__ == "__main__":
    if len(sys.argv) == 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py test-data train-data\n")
        sys.exit(1)

    inputs = sys.argv[1:]
    output = os.path.join("data", "stats")

    if not os.path.exists(output):
        os.makedirs(output)

    ds = {}
    for file_path in tqdm(inputs):
        name = os.path.basename(file_path).split(".")[0]
        steps = {
            "load data": {"fn": load_data},
            "test invalid data": {"fn": test_invalid_data},
            "test missing data": {"fn": test_missing_values},
            "save stats": {"fn": save_stats, "kwds": {"name": name, "path": output}},
            "save max load": {
                "fn": save_max_load,
                "kwds": {"name": name, "path": output},
            },
        }
        input = file_path
        it = tqdm(steps.items())
        for n, d in it:
            it.set_description(n)
            fn = d["fn"]
            data = fn(input, *d.get("args", []), **d.get("kwds", {}))
            input = data
        ds[name] = data

    test_information_leakage(ds["train"], ds["test"])
