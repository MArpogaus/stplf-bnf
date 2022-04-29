# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : preprocessing.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-03-26 11:32:07 (Marcel Arpogaus)
# changed : 2021-04-22 18:32:56 (Marcel Arpogaus)
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
    files = [
        file
        for file in os.listdir(data_path)
        if file.startswith("File") and file.endswith(".txt")
    ]
    if len(files) == 0:
        sys.stderr.write("dataset not found")
        sys.exit(1)

    file_name = os.path.join(data_path, "SME and Residential allocations.txt")
    df_categories = pd.read_csv(
        file_name, dtype={"Id": "uint16", "Code": "uint8"}, sep="\s+"
    )
    df_categories.columns = df_categories.columns.str.lower()

    residential_ids = df_categories[df_categories.code == 1].id

    data = dict()
    it = tqdm(files)
    for f in it:
        it.set_description(f"reading file: {f}")

        file_name = os.path.join(data_path, f)
        df = pd.read_csv(
            file_name,
            header=None,
            names=["id", "date_time", "load"],
            dtype={"id": "uint16", "load": "float32"},
            # index_col=["id","date_time"],
            # parse_dates=[1],
            # date_parser=date_parser,
            sep="\s+",
        )

        # drop all *non residential* ids
        it.set_description("select residential ids")
        df = df.loc[df.id.isin(residential_ids)]

        # parse timestamp
        it.set_description("generating timestamps")
        # digit 1-3 (day 1 = 2019/1/1)
        day_code = df.date_time // 100 - 1
        # digit 4-5 (1 = 00:00:00 - 00:29:59)
        time_code = df.date_time % 100 - 1

        date = pd.to_datetime(
            day_code,
            unit="D",
            origin=pd.Timestamp("2009-01-01"),
            infer_datetime_format=True,
        )
        time_delta = pd.to_timedelta(time_code * 30, unit="m")
        df.date_time = date + time_delta
        # df.date_time=(date.astype('int') // (10**9*60*60)).astype('uint32')
        # df.rename(columns={'date_time':'date'},inplace=True)
        # df['weekday']=date.dt.weekday

        # reduce mem footprint
        # it.set_description("reduce mem footprint")
        # df['minute']=time_code.astype('uint8')
        # df.id=df.id.astype('uint16')

        # replace all invalid (0) fileds with NaN
        # df.load.replace(0.0,np.nan,inplace=True)

        data[f] = df
    it.close()

    data = pd.concat(data.values())

    return data


# Date Cleaning ###############################################################
def clean_data(data):
    # Delete assets missing values
    size_ids = data.groupby("id").size()

    min_records = size_ids.max()
    low_data_ids = size_ids[size_ids != min_records].index
    tqdm.write(f"found {low_data_ids.unique().size} incomplete records")

    data = data[~data.id.isin(low_data_ids)]

    return data


# extract subset ##############################################################
def extract_subset(data, subset, seed):
    np.random.seed(seed)
    ids = list(sorted(data.id.unique()))
    np.random.shuffle(ids)
    ids = ids[: int(subset * len(ids))]
    return data[data.id.isin(ids)]


# time change #################################################################
#  * `25. Okt 2009` - Sommerzeit endete: $+1$ Stunde
#  * `28. Mär 2010` - Sommerzeit begann: $-1$ Stunde
#  * `31. Okt 2010` - Sommerzeit endete: $+1$ Stunde
def remove_time_change(data):
    it = tqdm(total=7)
    it.set_description("merge duplicated hours (summer time)")
    data = data.groupby(["id", "date_time"]).sum().reset_index()
    it.update()
    it.set_description("set date time index")
    data = data.set_index("date_time")  # set date time index to interpolate on
    it.update()
    it.set_description("groupby id")
    data = data.groupby("id")  # group by housholds
    it.update()
    it.set_description("resampling")
    data = data.resample("30T")  # resample with freq = 30M
    it.update()
    data = data.asfreq()  # Return the values at the new freq, essentially a reindex.
    it.update()
    it.set_description("interpolating")
    data = data.interpolate(
        method="linear", limit_direction="forward", limit=2
    )  # remove nans by forward interpolation
    it.update()
    it.set_description("dropping id")
    data = data.drop(
        "id", axis="columns"
    )  # drop id since interpolation turned it into garbage
    it.update()
    it.close()
    return data.reset_index()


# save_data ###################################################################
def save_data(data, path):
    data.to_csv(path, float_format="%.3f", index=False)


# main ########################################################################
if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(f"\tpython {sys.argv[0]} data-path\n")
        sys.exit(1)

    input = sys.argv[1]
    output = os.path.join("data", "prepared", "data.csv")

    params = yaml.safe_load(open("scripts/params.yaml"))["prepare"]

    dirname = os.path.dirname(output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    steps = {"load data": {"fn": load_data}, "clean data": {"fn": clean_data}}
    if "subset" in params.keys():
        steps.update(
            {
                "extract subset": {
                    "fn": extract_subset,
                    "kwds": {"subset": params["subset"], "seed": params["seed"]},
                }
            }
        )
    steps.update(
        {
            "fix time change": {"fn": remove_time_change},
            "save data": {"fn": save_data, "kwds": {"path": output}},
        }
    )
    it = tqdm(steps.items())
    for n, d in it:
        it.set_description(n)
        fn = d["fn"]
        output = fn(input, *d.get("args", []), **d.get("kwds", {}))
        input = output
