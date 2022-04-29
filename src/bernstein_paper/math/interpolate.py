# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : interpolate.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-20 10:49:40 (Marcel Arpogaus)
# changed : 2022-01-20 10:54:37 (Marcel Arpogaus)
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
import numpy as np
import tensorflow as tf
from tensorflow_probability.python.internal import (
    dtype_util,
    tensorshape_util,
)


def interp1d(x, y, q, fill_value_below=np.nan, fill_value_above=np.nan):
    dtype = dtype_util.common_dtype([x, y], dtype_hint=tf.float32)

    x = tf.convert_to_tensor(x, dtype_hint=dtype)
    x = tf.sort(x, axis=-1)
    y = tf.convert_to_tensor(y, dtype_hint=dtype)
    q = tf.convert_to_tensor(q, dtype_hint=dtype)

    y_rank = tensorshape_util.rank(y.shape)

    x_min = tf.math.reduce_min(x, axis=-1)
    x_max = tf.math.reduce_max(x, axis=-1)

    ny = y.shape[-1]

    q_below = q <= x_min[..., None]
    q_above = q >= x_max[..., None]

    # Find the index of the interval containing query

    idx = tf.searchsorted(x, q, side="right")

    idx_above = tf.minimum(idx, ny - 1)
    idx_below = tf.maximum(idx - 1, 0)

    # Interpolate
    x1 = tf.gather(x, idx_below, batch_dims=-1)
    x2 = tf.gather(x, idx_above, batch_dims=-1)

    y1 = tf.gather(y, idx_below, batch_dims=y_rank - 1)
    y2 = tf.gather(y, idx_above, batch_dims=y_rank - 1)

    alpha = (q - x1) / (x2 - x1)
    res = alpha * y2 + (1 - alpha) * y1
    res = tf.where(q_below, fill_value_below, res)
    res = tf.where(q_above, fill_value_above, res)
    return res
