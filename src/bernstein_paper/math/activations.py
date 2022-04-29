# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-12-06 19:12:20 (Marcel Arpogaus)
# changed : 2021-03-26 11:48:25 (Marcel Arpogaus)
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


def cumsum_fn(x: tf.Tensor, fn=tf.math.softplus) -> tf.Tensor:
    """Calculates y_0 = x_0, y_k = y_{k-1} + fn(y_k)

    :param x: input tensor
    :type x: tf.Tensor
    :param fn: function to apply. (Default: softmax)
    :type fn: Callable
    :returns:

    """
    y = tf.concat(
        (
            tf.zeros_like(x[..., :1]),
            x[..., :1],
            fn(x[..., 1:]),
        ),
        axis=-1,
    )
    return tf.cumsum(y[..., 1:], axis=-1)
