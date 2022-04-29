# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : empirical.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-03-12 17:04:27 (Marcel Arpogaus)
# changed : 2022-04-29 13:29:31 (Marcel Arpogaus)
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
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import dtype_util, tensor_util


class Empirical(tfd.Empirical):
    def __init__(self, samples, name="EmpiricalDistribution"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([samples], tf.float32)
            samples = tensor_util.convert_nonref_to_tensor(
                samples, name="pvector", dtype=dtype
            )
            super().__init__(samples=samples, name=name, event_ndims=0)

        # Ensure that the subclass (not base class) parameters are stored.
        self._parameters = parameters

    def _prob(self, x):
        dx = 1e-2
        dy = self.cdf(x + dx) - self.cdf(x - dx) + dtype_util.eps(self.dtype)
        return dy / 2.0 / dx

    @classmethod
    def _parameter_properties(cls, dtype=None, num_classes=None):
        return dict(
            # Annotations may optionally specify properties, such as `event_ndims`,
            # `default_constraining_bijector_fn`, `specifies_shape`, etc.; see
            # the `ParameterProperties` documentation for details.
            samples=tfp.util.ParameterProperties(event_ndims=1),
        )
