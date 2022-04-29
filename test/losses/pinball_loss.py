# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : pinball_loss.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-12-10 17:57:58 (Marcel Arpogaus)
# changed : 2021-12-13 12:18:35 (Marcel Arpogaus)
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
from bernstein_paper.distributions import (
    MixedNormal,
    NormalDistribution,
    QuantileRegressionDistributionWrapper,
)
from bernstein_paper.losses import PinballLoss
from tensorflow_probability.python.internal import test_util

tf.random.set_seed(42)
min_quantile_level = 0.01
num_quantiles = 99


@test_util.test_all_tf_execution_regimes
class ContinuousRankedProbabilityScoreTest(tf.test.TestCase):
    def test_0pc_50pc_100pc(self):
        y_true = tf.constant(
            [1, 9, 2, -5, -2, 6], shape=(2, 3), dtype=tf.dtypes.float32
        )
        y_pred = tf.constant([4, 8, 12, 8, 1, 3], shape=(2, 3), dtype=tf.dtypes.float32)
        y_pred = tf.repeat(y_pred[..., None], 3, axis=-1)
        pl = PinballLoss(min_quantile_level=0.0, constrain_quantiles=tf.identity)
        loss = pl(y_true, y_pred)

        self.assertAllClose(loss, tf.reduce_sum([4.8333, 2.75, 0.6666]), atol=1e-4)
