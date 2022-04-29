# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test_mean_quantile_score.py
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
from tensorflow_probability.python.internal import test_util
from bernstein_flow.bijectors import BernsteinBijectorLinearExtrapolate
from bernstein_flow.distributions import BernsteinFlow
from bernstein_flow.util import gen_flow
from bernstein_paper.distributions import (
    MixedNormal,
    NormalDistribution,
    QuantileRegressionDistributionWrapper,
)
from bernstein_paper.metrics import MeanQuantileScore
from bernstein_paper.losses import PinballLoss

tf.random.set_seed(42)
min_quantile_level = 0.01
num_quantiles = 99


def score(x, dist_class, pv):
    score = MeanQuantileScore(
        dist_class,
        min_quantile_level=min_quantile_level,
        num_quantiles=num_quantiles,
    )
    score.reset_state()
    score.update_state(x, pv)
    return score.result()


# @test_util.test_all_tf_execution_regimes
class MeanQuantileScoreTest(tf.test.TestCase):
    def test_normal(self):
        batch_size = 10

        pv = tf.random.uniform((batch_size, 2), -10, 10)
        dist = NormalDistribution(pv)
        x = dist.sample()

        result = score(x, NormalDistribution, pv)

    def test_normal_qdw(self):
        batch_size = 10

        pv = tf.random.uniform((batch_size, 10, 2), -5, 5)
        dist = NormalDistribution(pv)
        y_true = dist.sample()
        result_norm = score(y_true, NormalDistribution, pv)

        p = tf.linspace(min_quantile_level, 1 - min_quantile_level, num_quantiles)[
            ..., None, None
        ]
        q = dist.quantile(p)
        perm = tf.concat((tf.range(1, tf.rank(q)), [0]), 0)
        qdw_pv = tf.transpose(q, perm)
        result_qdw = score(
            y_true,
            lambda pv: QuantileRegressionDistributionWrapper(
                pv,
                min_quantile_level=min_quantile_level,
                constrain_quantiles=tf.identity,
            ),
            qdw_pv,
        )

        pl = PinballLoss(
            min_quantile_level=min_quantile_level, constrain_quantiles=tf.identity
        )
        pl_result = pl(y_true, qdw_pv)

        self.assertAllInRange(result_qdw, 0.0, dist.dtype.max)
        self.assertAllClose(result_norm, result_qdw)
        self.assertAllClose(result_qdw, pl_result)

    def test_mixed_normal(self):
        batch_size = 1
        pv = tf.random.uniform((batch_size, 2, 3), -10, 10)
        dist = MixedNormal(pv)
        x = dist.sample()
        result = score(x, MixedNormal, pv)

        self.assertAllInRange(result, 0.0, dist.dtype.max)

    def test_bernstein_flow(self):
        batch_size = 1
        pv = tf.random.uniform((batch_size, 10), -5, 5)
        kwds = dict(
            bb_class=BernsteinBijectorLinearExtrapolate,
            allow_values_outside_support=True,
            clip_to_bernstein_domain=False,
            scale_base_distribution=False,
        )
        dist = BernsteinFlow.from_pvector(pv, **kwds)

        x = dist.sample()
        result = score(x, gen_flow(**kwds), pv)

        self.assertAllInRange(result, 0.0, dist.dtype.max)
