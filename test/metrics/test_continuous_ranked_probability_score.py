# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test_continuous_ranked_probability_score.py
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
from bernstein_flow.bijectors import BernsteinBijectorLinearExtrapolate
from bernstein_flow.distributions import BernsteinFlow
from bernstein_flow.util import gen_flow
from bernstein_paper.distributions import (
    MixedNormal,
    NormalDistribution,
    QuantileRegressionDistributionWrapper,
)
from bernstein_paper.metrics import ContinuousRankedProbabilityScore
from properscoring import crps_gaussian, crps_quadrature
from tensorflow_probability.python.internal import test_util

tf.random.set_seed(42)


def score(x, dist_class, pv):
    score = ContinuousRankedProbabilityScore(dist_class)
    score.update_state(x, pv)
    return score.result()


def comp_crps_gaussian(x, dist):
    mu = dist.loc
    sig = dist.scale
    return tf.reduce_mean(crps_gaussian(x, mu, sig))


def comp_crps_quadrature(x, dist):
    return tf.reduce_mean(crps_quadrature(x, dist, tol=1e-4))


class ContinuousRankedProbabilityScoreTest(tf.test.TestCase):
    def test_normal(self):
        epochs = 10
        batch_size = 10
        for s in range(epochs):
            tf.random.set_seed(s)
            pv = tf.random.uniform((batch_size, 2), -10, 10)
            dist = NormalDistribution(pv)
            x = dist.sample()
            crps_result = score(x, NormalDistribution, pv)
            comp_result = comp_crps_gaussian(x, dist)
            self.assertAllClose(crps_result, comp_result, rtol=1e-4, atol=1e-4)

    def test_normal_qdw(self):
        epochs = 10
        batch_size = 10
        for s in range(epochs):
            tf.random.set_seed(s)
            pv = tf.random.uniform((batch_size, 10, 2), -5, 5)
            dist = NormalDistribution(pv)
            x = dist.sample()
            tol = 1e-7
            p = tf.linspace(tol, 1 - tol, 100)[..., None, None]
            q = dist.quantile(p)
            perm = tf.concat((tf.range(1, tf.rank(q)), [0]), 0)
            qdw_pv = tf.transpose(q, perm)
            crps_result = score(
                x,
                lambda pv: QuantileRegressionDistributionWrapper(
                    pv, constrain_quantiles=tf.identity
                ),
                qdw_pv,
            )
            comp_result = comp_crps_gaussian(x, dist)
            self.assertAllClose(crps_result, comp_result, rtol=1e-3, atol=1e-3)

    def test_mixed_normal(self):
        epochs = 10
        batch_size = 1
        for s in range(epochs):
            tf.random.set_seed(s)
            pv = tf.random.uniform((batch_size, 2, 3), -10, 10)
            dist = MixedNormal(pv)
            x = dist.sample()
            crps_result = score(x, MixedNormal, pv)
            comp_result = comp_crps_quadrature(x, dist)
            self.assertAllClose(crps_result, comp_result, rtol=1e-6, atol=1e-6)

    def test_bernstein_flow(self):
        epochs = 10
        batch_size = 1
        for s in range(epochs):
            tf.random.set_seed(s)
            pv = tf.random.uniform((batch_size, 10), -5, 5)
            kwds = dict(
                bb_class=BernsteinBijectorLinearExtrapolate,
                allow_values_outside_support=True,
                clip_to_bernstein_domain=False,
                scale_base_distribution=False,
            )
            dist = BernsteinFlow.from_pvector(pv, **kwds)

            x = dist.sample()
            crps_result = score(x, gen_flow(**kwds), pv)
            comp_result = comp_crps_quadrature(x, dist)
            self.assertAllClose(crps_result, comp_result, rtol=1e-6, atol=1e-5)
