# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : windowed_time_series_pipeline.py
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

# REQUIRED PYTHON MODULES #####################################################
import tensorflow as tf


class PatchGenerator:
    def __init__(self, window_size, shift):
        self.window_size = window_size
        self.shift = shift

    def __call__(self, load_data):
        def sub_to_patch(sub):
            return sub.batch(self.window_size, drop_remainder=True)

        data_set = tf.data.Dataset.from_tensor_slices(load_data)

        data_set = data_set.window(
            size=self.window_size,
            shift=self.shift,
            # stride=1,
            drop_remainder=True,
        )
        data_set = data_set.flat_map(sub_to_patch)
        # data_set = data_set.map(process_patch)

        return data_set


class BatchPreprocessor:
    def __init__(
        self,
        history_size,
        history_columns,
        meta_columns,
        prediction_columns,
        column_transformers={},
    ):
        self.history_size = history_size
        self.history_columns = history_columns
        self.meta_columns = meta_columns
        self.prediction_columns = prediction_columns
        self.column_transformers = column_transformers

        columns = sorted(list(set(history_columns + prediction_columns + meta_columns)))
        self.column_idx = {c: i for i, c in enumerate(columns)}

    def __call__(self, batch):
        y = []
        x_hist = []
        x_meta = []

        x_columns = sorted(set(self.history_columns + self.meta_columns))
        y_columns = sorted(self.prediction_columns)

        for c in y_columns:
            column = batch[:, self.history_size :, self.column_idx[c]]
            column = self.get_column_transformer(c)(column)
            y.append(column)

        if len(x_columns) == 0:
            ValueError("No feature columns provided")

        for c in x_columns:
            column = batch[:, :, self.column_idx[c], None]
            column = self.get_column_transformer(c)(column)
            if c in self.history_columns:
                x_hist.append(column[:, : self.history_size, 0])
            if c in self.meta_columns:
                x_meta.append(column[:, self.history_size + 1, None, ...])

        y = tf.stack(y, axis=2)
        x_hist = tf.stack(x_hist, axis=2)
        x_meta = tf.concat(x_meta, axis=2)

        return (x_hist, x_meta), y

    def get_column_transformer(self, column):
        return self.column_transformers.get(column, tf.identity)


class WindowedTimeSeriesPipeline:
    def __init__(
        self,
        history_size,
        prediction_size,
        history_columns,
        meta_columns,
        prediction_columns,
        shift,
        batch_size,
        cycle_length,
        shuffle_buffer_size,
        seed,
        column_transformers={},
    ):
        self.history_size = history_size
        self.prediction_size = prediction_size
        self.window_size = history_size + prediction_size
        self.history_columns = history_columns
        self.meta_columns = meta_columns
        self.prediction_columns = prediction_columns
        self.shift = shift
        self.batch_size = batch_size
        self.cycle_length = cycle_length
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        self.column_transformers = column_transformers

    def __call__(self, ds):

        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(
                self.cycle_length * self.shuffle_buffer_size, seed=self.seed
            )

        ds = ds.interleave(
            PatchGenerator(self.window_size, self.shift),
            cycle_length=self.cycle_length,
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        if self.shuffle_buffer_size > 0:
            ds = ds.shuffle(self.batch_size * self.shuffle_buffer_size, seed=self.seed)

        ds = ds.batch(self.batch_size)

        ds = ds.map(
            BatchPreprocessor(
                self.history_size,
                self.history_columns,
                self.meta_columns,
                self.prediction_columns,
                self.column_transformers,
            ),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds
