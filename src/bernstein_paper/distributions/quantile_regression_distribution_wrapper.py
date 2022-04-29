# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : quantile_regression_distribution_wrapper.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-07-29 17:57:39 (Marcel Arpogaus)
# changed : 2021-12-16 09:54:05 (Marcel Arpogaus)
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
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import (
    dtype_util,
    prefer_static,
    reparameterization,
    tensor_util,
    tensorshape_util,
)

from bernstein_paper.math.interpolate import interp1d
from bernstein_paper.math.activations import cumsum_fn


class QuantileRegressionDistributionWrapper(tfd.Distribution):
    def __init__(
        self,
        quantiles,
        min_quantile_level=None,
        max_quantile_level=None,
        constrain_quantiles=cumsum_fn,
        validate_args=False,
        allow_nan_stats=True,
        name="QuantileDistributionWrapper",
    ):

        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([quantiles], dtype_hint=tf.float32)

            self.quantiles = tensor_util.convert_nonref_to_tensor(
                quantiles, dtype=dtype, name="quantiles"
            )

            if min_quantile_level is None:
                self.min_quantile_level = dtype_util.eps(dtype)
            else:
                self.min_quantile_level = min_quantile_level
            self.min_quantile_level = tensor_util.convert_nonref_to_tensor(
                self.min_quantile_level, dtype=dtype, name="min_quantile_level"
            )

            if max_quantile_level is None:
                self.max_quantile_level = 1.0 - self.min_quantile_level
            else:
                self.max_quantile_level = max_quantile_level
            self.max_quantile_level = tensor_util.convert_nonref_to_tensor(
                self.max_quantile_level, dtype=dtype, name="max_quantile_level"
            )

            self.quantiles = constrain_quantiles(self.quantiles)

            super().__init__(
                dtype=dtype,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=name,
            )

    def reshape_out(self, sample_shape, y):
        output_shape = prefer_static.broadcast_shape(sample_shape, self.batch_shape)
        return tf.reshape(y, output_shape)

    def _batch_shape(self):
        shape = tf.TensorShape(prefer_static.shape(self.quantiles)[:-1])
        return tf.broadcast_static_shape(shape, tf.TensorShape([1]))

    def _event_shape(self):
        return tf.TensorShape([])

    def _log_prob(self, x):
        return tf.math.log(self.prob(x))

    def _prob(self, x):
        x_shape = prefer_static.shape(x)
        dx = 1e-2
        dy = self.cdf(x + dx) - self.cdf(x - dx) + dtype_util.eps(self.dtype)
        return self.reshape_out(x_shape, dy / 2.0 / dx)

    def _log_cdf(self, x):
        return tf.math.log(self.cdf(x))

    @tf.function
    def _cdf(self, x):
        batch_rank = tensorshape_util.rank(self.batch_shape)
        x_shape = prefer_static.shape(x)
        output_shape = prefer_static.broadcast_shape(x_shape, self.batch_shape)
        q = self.quantiles
        y = tf.linspace(self.min_quantile_level, self.max_quantile_level, q.shape[-1])

        # broadcast to final shape
        x = tf.broadcast_to(x, output_shape)
        sample_shape = x_shape[:-batch_rank]
        sample_rank = tf.rank(sample_shape)
        # if x in the shape form [S,BE], where S is sample_shape and BE is batch_shape,
        # we need to transpose it to [BE, S]
        needs_transpose = sample_rank > 0 and tf.reduce_any(
            x_shape[:batch_rank] != self.batch_shape
        )
        if needs_transpose:
            perm = tf.concat(
                [
                    tf.range(sample_rank, batch_rank + sample_rank),
                    tf.range(0, sample_rank),
                ],
                0,
            )
            x = tf.transpose(x, perm)  # [B, S]
        elif tf.reduce_all(x_shape[:batch_rank] == self.batch_shape):
            x = x[
                ..., tf.newaxis
            ]  # tf.reshape(x, self.batch_shape.as_list() + [-1])  # [B, 1]

        res = interp1d(
            q,
            y,
            x,
            tf.convert_to_tensor(0.0, dtype=self.dtype),
            tf.convert_to_tensor(1.0, dtype=self.dtype),
        )

        if needs_transpose:
            perm = tf.concat(
                [
                    tf.range(batch_rank, batch_rank + sample_rank),
                    tf.range(0, batch_rank),
                ],
                0,
            )
            res = tf.transpose(res, perm)

        return self.reshape_out(x_shape, res)

    def _quantile(self, p):
        input_shape = prefer_static.shape(p)
        q = tfp.math.batch_interp_regular_1d_grid(
            p[..., None],
            x_ref_min=self.min_quantile_level,
            x_ref_max=self.max_quantile_level,
            y_ref=self.quantiles,
            fill_value_below=self.dtype.min,
            fill_value_above=self.dtype.max,
            axis=-1,
        )

        return self.reshape_out(input_shape, q)
