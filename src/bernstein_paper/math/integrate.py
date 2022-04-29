# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : integrate.py
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
from tensorflow_probability.python.internal import (
    dtype_util,
    tensor_util,
    tensorshape_util,
)


def h(n, a, b):
    n = tf.cast(n, dtype=a.dtype)
    return (b - a) / 2 ** n


def print_row(n, row, h):
    s = tf.strings.format("{} {}", (2 ** (n - 1), h * 2))
    for i in range(n):
        s += tf.strings.format(" {}", row.read(i))
    tf.print(s)


def romberg(f, a, b, args=(), tol=1e-8, rtol=1e-8, max_n=15, show=False):
    with tf.name_scope("rombergs_integral"):
        dtype = dtype_util.common_dtype([a, b], tf.float64)
        a = tensor_util.convert_nonref_to_tensor(a, name="a", dtype=dtype)
        b = tensor_util.convert_nonref_to_tensor(b, name="b", dtype=dtype)
        tol = tensor_util.convert_nonref_to_tensor(tol, name="tol", dtype=dtype)
        rtol = tensor_util.convert_nonref_to_tensor(rtol, name="rtol", dtype=dtype)
        max_n = tensor_util.convert_nonref_to_tensor(
            max_n, name="max_n", dtype=tf.int32
        )
        n = tensor_util.convert_nonref_to_tensor(1, name="n", dtype=tf.int32)

        row = tf.TensorArray(dtype, size=1, clear_after_read=False)

        h_1 = h(1, a, b)
        row = row.write(0, h_1 * (f(a, *args) + f(b, *args)))
        last_result = row.read(0) + dtype.max

        def cond(n, row, last_result, h, *unused):
            if show:
                print_row(n, row, h)

            result = row.read(n - 1)

            err = tf.abs(result - last_result)
            rel_err = err / tf.abs(result)
            c = tf.reduce_all(tf.logical_or(err < tol, rel_err < rtol))

            return ~c

        def body(n, row, last_result, h, a, b):
            last_result = row.read(n - 1)

            last_row = row
            row = tf.TensorArray(last_row.dtype, size=n + 1, clear_after_read=False)

            rank = tensorshape_util.rank(a.shape)
            r = 2 * tf.range(2 ** (n - 1), dtype=last_row.dtype) + 1
            r_shape = [...] + [tf.newaxis] * rank
            x = a + r[r_shape] * h

            result = 0.5 * last_row.read(0) + h * tf.reduce_sum(f(x, *args), axis=0)
            row = row.write(0, result)

            for m in tf.range(1, n + 1):
                prev = row.read(m - 1)
                last = last_row.read(m - 1)
                result = prev + (prev - last) / (4 ** tf.cast(m, result.dtype) - 1)
                row = row.write(m, result)

            last_row.close()

            return n + 1, row, last_result, h / 2, a, b

        n, row, last_result, _, _, _ = tf.while_loop(
            cond,
            body,
            (n, row, last_result, h_1, a, b),
            maximum_iterations=max_n,
        )

        result = row.read(n - 1)
        row = row.close()

        return result, tf.abs(result - last_result), n
