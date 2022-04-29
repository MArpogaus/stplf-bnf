# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : train.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-06-22 17:05:38 (Marcel Arpogaus)
# changed : 2022-02-15 10:49:04 (Marcel Arpogaus)
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
# REQUIRED PYTHON MODULES #####################################################

import os
import argparse
import pathlib
import tfexp
import yaml

from bernstein_paper.util.config import get_cfg

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

    results_path = args.results_path
    results_path.mkdir(parents=True, exist_ok=True)

    test_mode = (
        args.test_mode or yaml.safe_load(open("scripts/params.yaml"))["test_mode"]
    )

    data_cfg.update(
        {
            "splits": ["train", "validate"],
            "seed": exp_cfg["seed"],
            "test_mode": test_mode,
        }
    )
    if test_mode:
        exp_cfg["epochs"] = 1

    command = "fit"

    # tf.debugging.set_log_device_placement(True)
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

    tfexp.fit(cfg)
