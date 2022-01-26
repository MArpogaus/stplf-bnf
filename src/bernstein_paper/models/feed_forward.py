# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : feed_forward.py
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

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model


def build_model(
    history_shape,
    meta_shape,
    output_shape,
    hidden_layers=[dict(units=50, activation="elu", kernel_initializer="he_normal")],
    output_layer_kwds=dict(activation="linear"),
    batch_normalization=True,
    name=None,
):

    hist_in = Input(shape=history_shape)
    meta_in = Input(shape=meta_shape)

    x1 = Flatten()(hist_in)
    x2 = Flatten()(meta_in)

    if batch_normalization:
        x1 = BatchNormalization()(x1)
        x2 = BatchNormalization()(x2)

    x = Concatenate()([x1, x2])

    for kwds in hidden_layers:
        x = Dense(**kwds)(x)
        if batch_normalization:
            x = BatchNormalization()(x)

    x = Dense(np.prod(output_shape), **output_layer_kwds)(x)
    x = Reshape(output_shape)(x)

    return Model(inputs=[hist_in, meta_in], outputs=x, name=name)
