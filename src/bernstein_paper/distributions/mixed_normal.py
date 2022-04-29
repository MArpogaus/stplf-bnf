# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : mixed_normal.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-07-29 17:57:39 (Marcel Arpogaus)
# changed : 2021-09-21 17:29:52 (Marcel Arpogaus)
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
from tensorflow_probability.python.internal import dtype_util, tensor_util


class MixedNormal(tfd.MixtureSameFamily):
    def __init__(self, pvector, name="MixedNormal"):
        parameters = dict(locals())
        with tf.name_scope(name) as name:
            dtype = dtype_util.common_dtype([pvector], tf.float32)
            pvector = tensor_util.convert_nonref_to_tensor(
                pvector, name="pvector", dtype=dtype
            )

            logits = pvector[..., 0]
            locs = pvector[..., 1]
            scales = tf.math.softplus(pvector[..., 2])

            super().__init__(
                mixture_distribution=tfd.Categorical(logits=logits),
                components_distribution=tfd.Normal(loc=locs, scale=scales),
                name="MixedNormal",
            )
        # Ensure that the subclass (not base class) parameters are stored.
        self._parameters = parameters

    @classmethod
    def _parameter_properties(cls, dtype=None, num_classes=None):
        return dict(
            # Annotations may optionally specify properties, such as `event_ndims`,
            # `default_constraining_bijector_fn`, `specifies_shape`, etc.; see
            # the `ParameterProperties` documentation for details.
            pvector=tfp.util.ParameterProperties(event_ndims=2),
        )
