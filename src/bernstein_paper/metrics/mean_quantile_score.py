# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : quantle_score.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-12-03 16:23:04 (Marcel Arpogaus)
# changed : 2022-02-25 15:36:48 (Marcel Arpogaus)
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
# REQUIRED MODULES #############################################################
import tensorflow as tf
from bernstein_paper.math.losses import pinball_loss
from bernstein_paper.util import find_quantile
from tensorflow_probability.python.internal import dtype_util, tensor_util


class MeanQuantileScore(tf.keras.metrics.Mean):
    def __init__(
        self,
        distribution_class,
        min_quantile_level=None,
        max_quantile_level=None,
        num_quantiles=100,
        name="mean_quantile_score",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.min_quantile_level = min_quantile_level
        self.max_quantile_level = max_quantile_level
        self.num_quantiles = num_quantiles

        self.distribution_class = distribution_class

    def update_state(self, y_true, pvector, sample_weight=None):
        dist = self.distribution_class(pvector)

        if self.min_quantile_level is None:
            min_quantile_level = dtype_util.eps(dist.dtype)
        else:
            min_quantile_level = self.min_quantile_level
        min_quantile_level = tensor_util.convert_nonref_to_tensor(
            min_quantile_level, dtype=dist.dtype, name="min_quantile_level"
        )

        if self.max_quantile_level is None:
            max_quantile_level = 1.0 - min_quantile_level
        else:
            max_quantile_level = self.max_quantile_level
        max_quantile_level = tensor_util.convert_nonref_to_tensor(
            max_quantile_level, dtype=dist.dtype, name="max_quantile_level"
        )

        y_true = tensor_util.convert_nonref_to_tensor(
            y_true, dtype=dist.dtype, name="y_true"
        )
        y_true = tf.squeeze(y_true)

        quantile_levels = tf.linspace(
            min_quantile_level, max_quantile_level, self.num_quantiles
        )

        dp = (max_quantile_level-min_quantile_level)/self.num_quantiles
        score = tf.zeros(dist.batch_shape)

        for q in quantile_levels:
            quantile = find_quantile(dist, q)
            score += pinball_loss(y_true, quantile, q)

        score *= dp * 2

        super().update_state(score, sample_weight)
