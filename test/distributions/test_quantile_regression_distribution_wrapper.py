# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test_quantile_regression_distribution_wrapper.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2022-01-20 10:49:40 (Marcel Arpogaus)
# changed : 2022-01-20 16:27:00 (Marcel Arpogaus)
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
from bernstein_paper.distributions import QuantileRegressionDistributionWrapper
from tensorflow_probability import distributions as tfd
from tensorflow_probability.python.internal import test_util

tf.random.set_seed(1)
min_quantile_level = 1e-4


def get_quantile_levels(batch_shape, min_quantile_level):
    shape = [...] + [None] * len(batch_shape)
    p = tf.linspace(min_quantile_level, 1 - min_quantile_level, 100)[shape]
    return p


@test_util.test_all_tf_execution_regimes
class QuantileRegressionDistributionWrapperTest(tf.test.TestCase):
    def gen_dist(self, batch_shape, dtype=tf.float32):
        if batch_shape == []:
            batch_shape = [1]
        n = tfd.Normal(
            loc=tf.random.uniform((batch_shape), -5, 5, dtype=dtype),
            scale=tf.random.uniform((batch_shape), 1, 5, dtype=dtype),
        )
        p = get_quantile_levels(batch_shape, min_quantile_level)
        q = n.quantile(p)
        perm = tf.concat((tf.range(1, tf.rank(q)), [0]), 0)
        pv = tf.transpose(q, perm)
        bs = QuantileRegressionDistributionWrapper(
            pv, min_quantile_level=min_quantile_level, constrain_quantiles=tf.identity
        )
        return n, bs

    def f(self, batch_shape):
        normal_dist, qr_dist = self.gen_dist(batch_shape)

        for input_shape in [
            [1],
            [1, 1],
            batch_shape,
            [1] + batch_shape,
            [10] + batch_shape,
        ]:
            # Check the distribution.
            self.assertEqual(normal_dist.batch_shape, qr_dist.batch_shape)
            self.assertEqual(normal_dist.event_shape, qr_dist.event_shape)
            p = get_quantile_levels(batch_shape, min_quantile_level)
            self.assertAllClose(
                normal_dist.quantile(p),
                qr_dist.quantile(p),
                atol=1e-5,
                rtol=1e-4,
            )

            mu = normal_dist.mean()
            self.assertAllClose(
                normal_dist.prob(mu), qr_dist.prob(mu), rtol=1e-4, atol=1e-4
            )
            self.assertAllClose(
                normal_dist.log_prob(mu), qr_dist.log_prob(mu), rtol=1e-4, atol=1e-4
            )
            p_min = 0.1
            min_q = normal_dist.quantile(p_min)
            max_q = normal_dist.quantile(1 - p_min)
            q = tf.linspace(min_q, max_q, 100)

            self.assertAllClose(
                normal_dist.cdf(q), qr_dist.cdf(q), rtol=1e-4, atol=1e-4
            )
            self.assertAllClose(
                normal_dist.log_cdf(q), qr_dist.log_cdf(q), rtol=1e-3, atol=1e-3
            )

    def test_dist(self):
        self.f(batch_shape=[])

    def test_dist_batch(self):
        self.f(batch_shape=[32])

    def test_dist_multi(self):
        self.f(batch_shape=[32, 48])
