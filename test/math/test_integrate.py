# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : test_integrate.py
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
import numpy as np
import tensorflow as tf
from bernstein_paper.math.integrate import romberg
from tensorflow_probability import distributions as tfd

tf.random.set_seed(42)


class RombergsMethodTest(tf.test.TestCase):
    # taken from scipy/scipy/integrate/tests/test_quadrature.py
    def test_romberg(self):
        # Typical function with two extra arguments:
        def myfunc(x, n, z):  # Bessel function integrand
            return tf.cos(n * x - z * tf.sin(x)) / np.pi

        val, _, _ = romberg(myfunc, 0, np.pi, args=(2, 1.8))
        table_val = 0.30614353532540296487
        self.assertAllClose(val, table_val, atol=1e-7)

    # taken from scipy/scipy/integrate/tests/test_quadrature.py
    def test_romberg_rtol(self):
        # Typical function with two extra arguments:
        def myfunc(x, n, z):  # Bessel function integrand
            return 1e19 * tf.cos(n * x - z * tf.sin(x)) / np.pi

        rtol = 1e-10
        val, _, _ = romberg(myfunc, 0, np.pi, args=(2, 1.8), rtol=rtol)
        table_val = 1e19 * 0.30614353532540296487
        self.assertAllClose(val, table_val, atol=1e-7, rtol=rtol)

    def test_batched_pdf(self):
        batch_shape = [32]
        tol = 1e-8
        dist = tfd.Normal(
            loc=tf.random.uniform(batch_shape, -10, 10, dtype=tf.float64), scale=1.0
        )

        val, _, _ = romberg(
            dist.prob,
            dist.quantile(tol),
            dist.quantile(1 - tol),
            tol=1e-8,
            rtol=1e-8,
            max_n=25,
        )
        self.assertAllClose(val, tf.ones(batch_shape))
