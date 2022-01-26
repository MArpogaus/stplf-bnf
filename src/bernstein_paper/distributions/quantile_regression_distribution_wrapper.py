# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : quantile_regression_distribution_wrapper.py
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
import numpy as np

import scipy.interpolate as spi

import tensorflow as tf

from tensorflow_probability import distributions as tfd

from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import tensor_util
from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.internal import prefer_static
from tensorflow_probability.python.internal import reparameterization

from ..losses import PinballLoss


class QuantileRegressionDistributionWrapper(tfd.Distribution):
    def __init__(
        self,
        quantiles,
        constrain_quantiles=PinballLoss.constrain_quantiles,
        validate_args=False,
        allow_nan_stats=True,
        name="QuantileDistributionWrapper",
    ):

        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([quantiles], dtype_hint=tf.float32)

            self.quantiles = tensor_util.convert_nonref_to_tensor(
                quantiles, dtype=dtype, name="quantiles"
            )

            assert self.quantiles.shape[-1] == 100, "100 Qunatiles reqired"

            self.quantiles = constrain_quantiles(self.quantiles)

            self._cdf_sp, self._quantile_sp = self.make_interp_spline()

            super().__init__(
                dtype=dtype,
                reparameterization_type=reparameterization.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                name=name,
            )

    def make_interp_spline(self):
        """
        Generates the Spline Interpolation.
        """
        quantiles = self.quantiles.numpy().copy()

        # Spline interpolation for cdf and dist
        x = quantiles
        x = x.reshape(-1, x.shape[-1])
        y = np.linspace(0.0, 1.0, 100, dtype=np.float32)

        x_min = np.min(x, axis=-1)
        x_max = np.max(x, axis=-1)

        cdf_sp = [
            spi.interp1d(y=np.squeeze(y), x=np.squeeze(x[i]), kind="linear")
            for i in range(x.shape[0])
        ]

        def cdf_sp_fn(x):
            y = []
            z_clip = np.clip(x, x_min, x_max)
            for i, ip in enumerate(cdf_sp):
                y.append(ip(z_clip[..., i]).astype(np.float32))
            y = np.stack(y, axis=-1)
            return y

        # linear interpolation for quantiles
        # clamp extreme values to value range of dtype
        float_min = np.finfo(np.float32).min * np.ones_like(quantiles[..., :1])
        float_max = np.finfo(np.float32).max * np.ones_like(quantiles[..., -1:])

        y = np.concatenate([float_min, quantiles, float_max], axis=-1)
        y = y.reshape(-1, y.shape[-1])

        tol = 1e-20
        percentiles = np.linspace(tol, 1.0 - tol, 100, dtype=np.float32)
        x = np.concatenate([np.zeros(1), percentiles, np.ones(1)], axis=-1)

        quantile_sp = [
            spi.interp1d(y=np.squeeze(y[i]), x=np.squeeze(x), kind="linear")
            for i in range(y.shape[0])
        ]

        def quantile_sp_fn(p):
            q = []
            p_clip = np.clip(p, np.zeros_like(x_min), np.ones_like(x_max))
            for i, ip in enumerate(quantile_sp):
                q.append(ip(p_clip[..., i]).astype(np.float32))
            q = np.stack(q, axis=-1)
            return q

        return cdf_sp_fn, quantile_sp_fn

    def reshape_out(self, sample_shape, y):
        output_shape = prefer_static.broadcast_shape(sample_shape, self.batch_shape)
        return tf.reshape(y, output_shape)

    def _eval_spline(self, x, attr):
        x = np.asarray(x, dtype=np.float32)
        batch_rank = tensorshape_util.rank(self.batch_shape)
        sample_shape = x.shape

        if x.shape[-batch_rank:] == self.batch_shape:
            shape = list(x.shape[:-batch_rank]) + [-1]
            x = tf.reshape(x, shape)
        else:
            x = x[..., None]

        return self.reshape_out(sample_shape, getattr(self, attr)(x))

    def _batch_shape(self):
        shape = tf.TensorShape(prefer_static.shape(self.quantiles)[:-1])
        return tf.broadcast_static_shape(shape, tf.TensorShape([1]))

    def _event_shape(self):
        return tf.TensorShape([])

    def _log_prob(self, x):
        return np.log(self.prob(x))

    def _prob(self, x, dx=1e-2):
        dy = self.cdf(x + dx) - self.cdf(x - dx)
        return self.reshape_out(x.shape, dy / 2.0 / dx)

    def _log_cdf(self, x):
        return np.log(self.cdf(x))

    def _cdf(self, x):
        return self._eval_spline(x, "_cdf_sp")

    def _mean(self):
        return self.quantile(0.5)

    def _quantile(self, p):
        # input_shape = p.shape
        # q = self.quantiles
        # perm = tf.concat([[q.ndim - 1], tf.range(0, q.ndim - 1)], 0)
        # q = tfp.math.interp_regular_1d_grid(
        #     p,
        #     x_ref_min=0.,
        #     x_ref_max=1.,
        #     y_ref=tf.transpose(q, perm),
        #     axis=0)
        # return self.reshape_out(input_shape, q)
        return self._eval_spline(p, "_quantile_sp")
