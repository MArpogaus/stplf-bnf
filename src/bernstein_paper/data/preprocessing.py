# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : preprocessing.py
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
class MyMinMaxScaler():
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range

    def fit(self, X, Y=None):
        self.min = X.min()
        self.max = X.max()

    def transform(self, X, Y=None):
        X_std = (X - self.min) / (self.max - self.min)
        X_scaled = X_std * \
            (self.feature_range[1] - self.feature_range[0]
             ) + self.feature_range[0]
        return X_scaled if Y is None else (X_scaled, self.transform(Y))

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X, Y)


class CERDataScaler():
    def __init__(self, load_col=0, weather_col=1):
        self.load_col = load_col
        self.weather_col = weather_col
        self.load_scaler = MyMinMaxScaler(feature_range=(0, 1))
        self.weather_scaler = MyMinMaxScaler(feature_range=(-1, 1))

    def fit(self, X, Y=None):
        load_data = X[:, :, self.load_col]
        weather_data = X[:, :, self.weather_col]

        self.load_scaler.fit(load_data)
        self.weather_scaler.fit(weather_data)

    def transform(self, X, Y=None):
        load_data = X[:, :, self.load_col]
        weather_data = X[:, :, self.weather_col]

        X[:, :, self.load_col] = self.load_scaler.transform(load_data)
        X[:, :, self.weather_col] = self.weather_scaler.transform(
            weather_data)
        if Y is not None:
            Y = self.load_scaler.transform(Y)

            return X, Y
        else:
            return X

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X, Y)
