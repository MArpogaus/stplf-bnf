# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : pinball_loss.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-07-29 17:57:39 (Marcel Arpogaus)
# changed : 2021-12-14 11:54:20 (Marcel Arpogaus)
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
import tensorflow as tf
from bernstein_paper.math.losses import pinball_loss
from bernstein_paper.math.activations import cumsum_fn
from tensorflow.keras.losses import Loss
from tensorflow_probability.python.internal import dtype_util, tensor_util


class PinballLoss(Loss):
    def __init__(
        self,
        min_quantile_level=None,
        max_quantile_level=None,
        constrain_quantiles=cumsum_fn,
        name="pinball_loss",
        **kwargs
    ):
        with tf.name_scope(name) as name:
            self.min_quantile_level = min_quantile_level
            self.max_quantile_level = max_quantile_level

            self.constrain_quantiles = constrain_quantiles
            super().__init__(name=name, **kwargs)

    def call(self, y, quantiles):
        """Calculate Pinball Loss on unconstrained ANN outputs
         The quantile levels are automatically derived from the last dim of `pvector`.

        :param y: Ground truth values.
        :type y: Tensor, shape = [batch_size, d0, .. dN]
        :param quantiles: Unconstrained Quantiles.
        :type quantiles: Tensor, shape = [batch_size, d0, .. dN, num_quantiles]
        :returns:

        """
        dtype = dtype_util.common_dtype([y, quantiles], dtype_hint=tf.float32)

        if self.min_quantile_level is None:
            min_quantile_level = dtype_util.eps(dtype)
        else:
            min_quantile_level = self.min_quantile_level
        min_quantile_level = tensor_util.convert_nonref_to_tensor(
            min_quantile_level, dtype=dtype, name="min_quantile_level"
        )

        if self.max_quantile_level is None:
            max_quantile_level = 1.0 - min_quantile_level
        else:
            max_quantile_level = self.max_quantile_level
        max_quantile_level = tensor_util.convert_nonref_to_tensor(
            max_quantile_level, dtype=dtype, name="max_quantile_level"
        )

        y = tensor_util.convert_nonref_to_tensor(y, dtype=dtype, name="y")
        y = tf.squeeze(y)
        quantiles = tensor_util.convert_nonref_to_tensor(
            quantiles, dtype=dtype, name="quantiles"
        )
        quantiles = self.constrain_quantiles(quantiles)

        num_quantiles = quantiles.shape[-1]

        quantile_levels = tf.linspace(
            min_quantile_level, max_quantile_level, num_quantiles
        )

        loss = tf.zeros([])
        for i in range(num_quantiles):
            loss += pinball_loss(y, quantiles[..., i], quantile_levels[i])

        return tf.reduce_mean(loss)
