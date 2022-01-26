# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : pinball_loss.py
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
import tensorflow as tf

from tensorflow.keras.losses import Loss


class PinballLoss(Loss):
    def __init__(
            self,
            **kwargs):

        super().__init__(**kwargs)

    @classmethod
    def constrain_quantiles(cls: type,
                            quantiles_unconstrained: tf.Tensor,
                            fn=tf.math.softplus) -> tf.Tensor:
        """
        Class method to calculate theta_1 = h_1, theta_k = theta_k-1 + exp(h_k)

        :param      cls:                  The class as implicit first argument.
        :type       cls:                  type
        :param      theta_unconstrained:  The unconstrained Bernstein
                                          coefficients.
        :type       theta_unconstrained:  Tensor
        :param      fn:                   The used activation function.
        :type       fn:                   Function

        :returns:   The constrained Bernstein coefficients.
        :rtype:     Tensor
        """
        q = tf.concat((tf.zeros_like(quantiles_unconstrained[..., :1]),
                       quantiles_unconstrained[..., :1],
                       fn(quantiles_unconstrained[..., 1:])), axis=-1)
        return tf.cumsum(q[..., 1:], axis=-1)

    @classmethod
    def slice_parameter_vectors(cls, pvector):
        num_qunantiles = pvector.shape[-1]
        sliced_pvectors = []
        for d in range(num_qunantiles):
            sliced_pvector = pvector[:, :, d, None]
            sliced_pvectors.append(sliced_pvector)
        return sliced_pvectors

    @classmethod
    def pinball_loss(cls, y_true, y_pred, q):
        ''' Pinball loss for Tensorflow Backend '''
        error = tf.subtract(y_true, y_pred)
        return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error), axis=-1)

    def call(self, y, pvector):
        pvector = self.constrain_quantiles(pvector)
        pvs = self.slice_parameter_vectors(pvector)

        losses = []
        for i, pv in enumerate(pvs):
            q = i / len(pvs)
            losses.append(self.pinball_loss(y, pv, q))

        return tf.reduce_mean(tf.add_n(losses))
