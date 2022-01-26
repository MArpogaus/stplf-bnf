# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : evaluate.py
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
import yaml
import sys
import tfexp


def evaluate(cfg_file, metrics_file):
    res_dict = tfexp.evaluate(cfg_file)
    res = next(iter(res_dict.values()))
    with open(metrics_file, 'w+') as f:
        yaml.dump(res["res"], f)


# main ########################################################################
if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Arguments error. Usage:\n")
        sys.stderr.write(f"\tpython {sys.argv[0]} config-file metrics-file\n")
        sys.exit(1)

    config_file = sys.argv[1]
    metrics_file = sys.argv[2]
    metrics_path = os.path.dirname(metrics_file)
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)

    evaluate(config_file, metrics_file)
