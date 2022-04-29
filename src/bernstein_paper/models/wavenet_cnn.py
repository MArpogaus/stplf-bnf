# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : wavenet_cnn.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-07-29 17:57:39 (Marcel Arpogaus)
# changed : 2021-07-29 18:02:20 (Marcel Arpogaus)
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

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import BatchNormalization


def build_model(history_shape,
                meta_shape,
                output_shape,
                conv_layers=[
                    dict(filters=20, kernel_size=2,
                         padding="causal",
                         activation="relu",
                         dilation_rate=rate)
                    for rate in (1, 2, 4, 8, 16, 32, 64)
                ],
                hidden_layers=[
                    dict(units=neurons,
                         activation='elu',
                         kernel_initializer="he_normal")
                    for neurons in (100, 100, 50)
                ],
                output_layer_kwds=dict(activation='linear'),
                batch_normalization=True,
                name=None):

    hist_in = Input(shape=history_shape)
    meta_in = Input(shape=meta_shape)

    hist_conv = hist_in
    for kwds in conv_layers:
        hist_conv = Conv1D(**kwds)(hist_conv)

    x1 = Flatten()(hist_conv)
    x2 = Flatten()(meta_in)

    x = Concatenate()([x1, x2])

    if batch_normalization:
        x = BatchNormalization()(x)

    for kwds in hidden_layers:
        x = Dense(**kwds)(x)
        if batch_normalization:
            x = BatchNormalization()(x)

    x = Dense(np.prod(output_shape), **output_layer_kwds)(x)
    x = Reshape(output_shape)(x)

    return Model(inputs=[hist_in, meta_in], outputs=x, name=name)
