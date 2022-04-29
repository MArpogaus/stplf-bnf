# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : continuous_ranked_probability_score.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-07-29 17:57:39 (Marcel Arpogaus)
# changed : 2021-12-16 11:32:16 (Marcel Arpogaus)
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
from bernstein_paper.math.integrate import romberg
from bernstein_paper.util import find_quantile
from tensorflow_probability.python.internal import dtype_util, tensor_util


@tf.function
def crps(dist, y_true, tol):
    with tf.name_scope("crps"):
        dtype = dtype_util.common_dtype([dist, y_true, tol], tf.float64)
        y_true = tensor_util.convert_nonref_to_tensor(y_true, name="b", dtype=dtype)
        tol = tensor_util.convert_nonref_to_tensor(tol, name="tol", dtype=dtype)

        cdf_error_msg = "CDF does not meet tolerance requirements at {} extreme(s)!"
        integration_warning_msg = "integration accuracy not achieved! ({} > {})"
        cdf = dist.cdf

        # Find bounds
        x_min = find_quantile(dist, tol / 10)
        p = cdf(x_min)
        tf.debugging.assert_less_equal(p, tol, message=cdf_error_msg.format("lower"))

        x_max = find_quantile(dist, 1 - (tol / 10))
        p = cdf(x_max)
        tf.debugging.assert_greater_equal(
            p, 1 - tol, message=cdf_error_msg.format("upper")
        )

        # CRPS = int_-inf^inf (F(y) - H(x))**2 dy
        #      = int_-inf^x F(y)**2 dy + int_x^inf (1 - F(y))**2 dy

        def lhs(x):
            # left hand side of CRPS integral
            return tf.square(cdf(x))

        def rhs(x):
            # right hand side of CRPS integral
            return tf.square(1.0 - cdf(x))

        lhs_int, err, n = romberg(
            lhs, x_min, y_true, tol=tol / 10, rtol=tol / 100, max_n=25
        )
        rel_err = err / tf.abs(lhs_int)
        err = tf.where(err < rel_err, err, rel_err)
        tf.debugging.assert_less_equal(
            err,
            tol,
            tf.strings.format(
                "lhs " + integration_warning_msg, (tf.reduce_max(err), tol)
            ),
        )

        rhs_int, err, n = romberg(
            rhs, y_true, x_max, tol=tol / 10, rtol=tol / 100, max_n=25
        )
        rel_err = err / tf.abs(rhs_int)
        err = tf.where(err < rel_err, err, rel_err)
        tf.debugging.assert_less_equal(
            err,
            tol,
            tf.strings.format(
                "rhs" + integration_warning_msg, (tf.reduce_max(err), tol)
            ),
        )

        score = lhs_int + rhs_int
        return score


class ContinuousRankedProbabilityScore(tf.keras.metrics.Mean):
    def __init__(
        self,
        distribution_class,
        name="continuous_ranked_probability_score",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.distribution_class = distribution_class
        self.tol = 1e-6

    def update_state(self, y_true, pvector, sample_weight=None):
        orig_dtype = y_true.dtype

        y_true = tf.squeeze(y_true)
        y_true = tf.cast(y_true, tf.float64)
        dist = self.distribution_class(tf.cast(pvector, tf.float64))

        score = crps(dist, y_true, self.tol)

        return super().update_state(
            tf.cast(score, orig_dtype), sample_weight=sample_weight
        )
