# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test_losses.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-12-03 17:18:23 (Marcel Arpogaus)
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

"""Tests for pinball loss."""

import numpy as np
import tensorflow as tf
from bernstein_paper.math.losses import pinball_loss


def test_all_correct_50pc():
    y_true = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3), dtype=tf.dtypes.float32)
    loss = tf.reduce_mean(pinball_loss(y_true, y_true, 0.5))
    assert loss == 0


def test_50pc():
    y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3), dtype=tf.dtypes.float32)
    y_pred = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3), dtype=tf.dtypes.float32)
    loss = tf.reduce_mean(pinball_loss(y_true, y_pred, 0.5))
    np.testing.assert_almost_equal(loss, 2.75, 3)


def test_quantile_0pc():
    y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3), dtype=tf.dtypes.float32)
    y_pred = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3), dtype=tf.dtypes.float32)
    loss = tf.reduce_mean(pinball_loss(y_true, y_pred, 0.0))
    np.testing.assert_almost_equal(loss, 4.8333, 3)


def test_quantile_10pc():
    y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3), dtype=tf.dtypes.float32)
    y_pred = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3), dtype=tf.dtypes.float32)
    loss = tf.reduce_mean(pinball_loss(y_true, y_pred, 0.1))
    np.testing.assert_almost_equal(loss, 4.4166, 3)


def test_quantile_90pc():
    y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3), dtype=tf.dtypes.float32)
    y_pred = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3), dtype=tf.dtypes.float32)
    loss = tf.reduce_mean(pinball_loss(y_true, y_pred, 0.9))
    np.testing.assert_almost_equal(loss, 1.0833, 3)


def test_quantile_100pc():
    y_true = tf.constant([1, 9, 2, -5, -2, 6], shape=(2, 3), dtype=tf.dtypes.float32)
    y_pred = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3), dtype=tf.dtypes.float32)
    loss = tf.reduce_mean(pinball_loss(y_true, y_pred, 1.0))
    np.testing.assert_almost_equal(loss, 0.6666, 3)
