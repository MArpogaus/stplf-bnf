# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : mean_squared_error.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-07-29 17:57:39 (Marcel Arpogaus)
# changed : 2021-10-20 13:11:57 (Marcel Arpogaus)
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
import tensorflow as tf
from bernstein_paper.util import find_quantile


class MedianSquaredError(tf.keras.metrics.MeanSquaredError):
    def __init__(
        self, distribution_class, name="median_squared_error", scale=1.0, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.distribution_class = distribution_class
        self.scale = scale

    def update_state(self, y_true, pvector, sample_weight=None):
        dist = self.distribution_class(pvector)
        mean = find_quantile(dist, 0.5)
        super().update_state(y_true * self.scale, mean * self.scale, sample_weight)
