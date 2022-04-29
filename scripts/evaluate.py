# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-04-22 13:56:01 (Marcel Arpogaus)
# changed : 2022-03-21 12:03:39 (Marcel Arpogaus)
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
# REQUIRED MODULES ############################################################
import argparse
import os
import pathlib

import tfexp
import yaml

from bernstein_paper.util.config import get_cfg


# FUNCTION DEFINITIONS ########################################################
def evaluate(cfg, metrics_file):
    res_dict = tfexp.evaluate(cfg)
    res = next(iter(res_dict.values()))
    with open(metrics_file, "w+") as f:
        yaml.dump(res["res"], f)


# MAIN ########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument(
        "exp_cfg", type=argparse.FileType("r"), help="Experiment configuration"
    )
    parser.add_argument(
        "data_cfg", type=argparse.FileType("r"), help="Data configuration"
    )
    parser.add_argument(
        "results_path",
        type=pathlib.Path,
        help="destination for model checkpoints and logs.",
    )
    parser.add_argument(
        "metrics_file",
        type=pathlib.Path,
        help="file to store metrics",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        required=False,
        help="enable some speed optimizations for testing",
    )

    args = parser.parse_args()
    exp_cfg = yaml.load(args.exp_cfg, Loader=yaml.Loader)
    data_cfg = yaml.safe_load(args.data_cfg)

    model_kwds_path = os.path.join(
        os.path.dirname(args.exp_cfg.name), "model_kwds.yaml"
    )
    model_kwds = yaml.safe_load(open(model_kwds_path))
    model_kwds["baseline"] = data_cfg

    results_path = args.results_path

    metrics_file = args.metrics_file
    metrics_file.parent.mkdir(parents=True, exist_ok=True)

    test_mode = (
        args.test_mode or yaml.safe_load(open("scripts/params.yaml"))["test_mode"]
    )

    data_cfg.update(
        {
            "splits": ["test"],
            "seed": exp_cfg["seed"],
            "test_mode": test_mode,
            "batch_size": 8,
        }
    )
    if test_mode:
        exp_cfg["epochs"] = 1

    command = "evaluate"

    # gpus = tf.config.list_logical_devices("GPU")
    # strategy = tf.distribute.MirroredStrategy(gpus)
    # with strategy.scope():
    cfg = get_cfg(
        command=command,
        results_path=results_path,
        model_kwds=model_kwds[exp_cfg["model"]],
        data_kwds=data_cfg,
        **exp_cfg
    )

    evaluate(cfg, metrics_file)
