# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : features.py
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

import yaml

import numpy as np

import pandas as pd

from tqdm import tqdm

from holidays import Ireland as holidays_ir
from wwo_hist import retrieve_hist_data


# helper functions
def dowload_weather_data(api_key, frequency, start_date, end_date):
    location_list = ["Dublin"]
    hist_weather_data = retrieve_hist_data(
        api_key, location_list, start_date, end_date, frequency, store_df=True
    )
    return hist_weather_data


# Load data ###################################################################
def load_data(data_path):
    data = pd.read_csv(
        data_path,
        parse_dates=["date_time"],
        infer_datetime_format=True,
        dtype={"id": "uint16", "load": "float32"},
    )
    return data


# features ####################################################################
def add_holiday(data):

    start_date = data.date_time.dt.date.min()
    end_date = data.date_time.dt.date.max()

    holidays = holidays_ir()

    h = holidays[start_date:end_date]
    is_holiday = np.uint8(np.isin(data.date_time.dt.date, h))

    data["is_holiday"] = is_holiday
    return data


def add_weekday(data):
    data["weekday"] = np.uint8(data.date_time.dt.weekday)
    return data


def add_weather(data, api_key):
    frequency = 1
    start_date = data.date_time.dt.date.min()
    end_date = data.date_time.dt.date.max()

    start_date_str = start_date.strftime("%d-%b-%Y").upper()
    end_date_str = end_date.strftime("%d-%b-%Y").upper()

    weather_data = dowload_weather_data(
        api_key, frequency, start_date_str, end_date_str
    )
    weather_data = weather_data[0]
    weather_data = weather_data[["date_time", "tempC"]]

    data = pd.merge_ordered(
        left=data,
        right=weather_data,
        on="date_time",
        fill_method="ffill",
        how="left",
    )
    return data


# save_data ###################################################################
def save_data(data, path):
    data.to_csv(path, float_format="%.3f", index=False)


# main ########################################################################
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write("\tpython prepare.py data-path\n")
        sys.exit(1)

    input = sys.argv[1]
    output = os.path.join("data", "features", "data.csv")

    params = yaml.safe_load(open("scripts/params.yaml"))["features"]

    dirname = os.path.dirname(output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    steps = {"load data": {"fn": load_data}}

    for f in params["generate"]:
        if isinstance(f, dict):
            f, kwds = next(iter(f.items()))
        else:
            kwds = {}
        fn = globals()["add_" + f]
        steps.update({"adding " + f: {"fn": fn, "kwds": kwds}})

    steps.update(
        {
            "save data": {"fn": save_data, "kwds": {"path": output}},
        }
    )

    it = tqdm(steps.items())
    for n, d in it:
        it.set_description(n)
        fn = d["fn"]
        output = fn(input, *d.get("args", []), **d.get("kwds", {}))
        input = output
