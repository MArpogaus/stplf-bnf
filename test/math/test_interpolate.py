# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test_interpolate.py
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
import tensorflow as tf
from bernstein_paper.math.interpolate import interp1d
import numpy as np

tf.random.set_seed(42)


class InterpolationTest(tf.test.TestCase):
    def test_interp1d(self):
        x = tf.linspace(0, 10, 100)
        y = 0.2 * x
        q = tf.random.uniform([10000], 0, 10, dtype=x.dtype)
        res = interp1d(x, y, q)
        self.assertAllClose(res, 0.2 * q, atol=1e-5)

    def test_interp1d_out_of_bound(self):
        x = tf.linspace(0, 10, 100)
        y = 1.1 * x
        q = tf.random.uniform([10000], -1, 11, dtype=x.dtype)
        res = interp1d(x, y, q)
        expected = tf.where((q <= 0) ^ (q >= 10), np.nan, 1.1 * q)
        self.assertAllClose(res, expected, atol=1e-5)
