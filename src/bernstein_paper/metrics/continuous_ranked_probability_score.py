# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : continuous_ranked_probability_score.py
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

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf

from absl import logging


@tf.function
def trapez(y, x):
    d = x[1:] - x[:-1]
    return tf.reduce_sum(d * (y[1:] + y[:-1]) / 2.0, axis=0)


class ContinuousRankedProbabilityScore(tf.keras.metrics.Mean):
    def __init__(
        self,
        distribution_class,
        name="continuous_ranked_probability_score",
        scale=1.0,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.distribution_class = distribution_class
        self.scale = scale
        self.tol = 1e-4

    def update_state(self, y_true, pvector, sample_weight=None):
        y_true = tf.squeeze(y_true)

        n_points = 10000

        dist = self.distribution_class(pvector)
        cdf = dist.cdf

        # Note that infinite values for xmin and xmax are valid, but
        # it slows down the resulting quadrature significantly.
        try:
            x_min = dist.quantile(self.tol)
            x_max = dist.quantile(1 - self.tol)
        except:
            x_min = -(10 ** (2 + y_true // 10))
            x_max = 10 ** (2 + y_true // 10)

        # make sure the bounds haven't clipped the cdf.
        warning = "CDF does not meet tolerance requirements at {} extreme(s)!"

        if tf.math.reduce_any(cdf(x_min) >= self.tol):
            logging.warning(warning.format("lower"))
        if tf.math.reduce_any(cdf(x_max) < (1.0 - self.tol)):
            logging.warning(warning.format("upper"))

            # CRPS = int_-inf^inf (F(y) - H(x))**2 dy
            #      = int_-inf^x F(y)**2 dy + int_x^inf (1 - F(y))**2 dy

        def lhs(x):
            # left hand side of CRPS integral
            return tf.square(cdf(x))

        def rhs(x):
            # right hand side of CRPS integral
            return tf.square(1.0 - cdf(x))

        lhs_x = tf.linspace(x_min, y_true, n_points)
        lhs_int = trapez(lhs(lhs_x), lhs_x)

        rhs_x = tf.linspace(y_true, x_max, n_points)
        rhs_int = trapez(rhs(rhs_x), rhs_x)

        score = lhs_int + rhs_int

        return super().update_state(score, sample_weight=sample_weight)
