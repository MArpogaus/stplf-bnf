# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : __init__.py
# author  : Marcel Arpogaus <marcel dot arpogaus at gmail dot com>
#
# created : 2021-07-26 14:53:27 (Marcel Arpogaus)
# changed : 2022-03-21 12:22:39 (Marcel Arpogaus)
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

import os
from functools import partial

import pandas as pd
import tensorflow as tf
from bernstein_flow.util import gen_flow
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tfexp.configuration import Configuration

from bernstein_paper.data.cer_data_loader import load_data
from bernstein_paper.distributions import (
    Empirical, MixedNormal, NormalDistribution,
    QuantileRegressionDistributionWrapper)
from bernstein_paper.losses import NegativeLogarithmicLikelihood
from bernstein_paper.metrics import (ContinuousRankedProbabilityScore,
                                     MeanQuantileScore, MedianAbsoluteError,
                                     MedianSquaredError)

min_quantile_level = 0.01


def get_distribution_class(distribution, distribution_kwds):
    if distribution == "quantile_regression":

        return QuantileRegressionDistributionWrapper
    elif distribution == "gaussian_mixture_model":

        return MixedNormal
    elif distribution == "normal_distribution":

        return NormalDistribution
    elif distribution == "bernstein_flow":

        return gen_flow(**distribution_kwds)
    elif distribution == "empirical":

        return Empirical
    else:
        raise ValueError(f'Model "{distribution}" unknown.')


def get_loss(distribution, distribution_kwds):
    if distribution == "quantile_regression":
        from bernstein_paper.losses import PinballLoss

        return PinballLoss(min_quantile_level=min_quantile_level)
    else:
        distribution_cls = get_distribution_class(distribution, distribution_kwds)
        return NegativeLogarithmicLikelihood(distribution_cls)


def get_model(
    architecture, history_size, meta_columns, prediction_size, output_shape, name
):
    if architecture == "feed_forward":
        from bernstein_paper.models.feed_forward import build_model

    elif architecture == "wavenet":
        from bernstein_paper.models.wavenet_cnn import build_model

    elif architecture == "baseline":
        from bernstein_paper.models.baseline import build_model

    else:
        raise ValueError(f"Architecture {architecture} unknown")

    return partial(
        build_model,
        history_shape=(history_size, 1),
        meta_shape=(1, len(meta_columns)),
        output_shape=[prediction_size] + output_shape,
        name=name,
    )


def get_callbacks(lr, results_path, verbose=True):
    lr_patience = 3
    max_lr_reductions = 3
    min_lr = lr * (0.1**max_lr_reductions)
    es_patience = lr_patience * 3 + 1
    return [
        ModelCheckpoint(
            os.path.join(results_path, "mcp/weights"),
            monitor="val_loss",
            mode="auto",
            verbose=verbose,
            save_weights_only=True,
            save_best_only=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="auto",
            patience=es_patience,
            restore_best_weights=True,
            verbose=verbose,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="auto",
            patience=lr_patience,
            factore=0.1,
            min_lr=min_lr,
            verbose=verbose,
        ),
        CSVLogger(os.path.join(results_path, "log.csv"), append=False),
    ]


def get_metrics(distribution_class, data_stats_path=None):
    if data_stats_path:
        data_stats_file = os.path.join(data_stats_path, "train.csv")
        data_stats = pd.read_csv(data_stats_file, index_col=0)
        load_max = data_stats.loc["max", "load"]
    else:
        load_max = 1

    kwds = dict(distribution_class=distribution_class)

    metrics = [
        m(**kwds, scale=load_max)
        for m in (
            MedianAbsoluteError,
            MedianSquaredError,
        )
    ] + [
        MeanQuantileScore(
            min_quantile_level=min_quantile_level, num_quantiles=99, **kwds
        ),
    ]
    if distribution_class is not Empirical:
        metrics += [
            ContinuousRankedProbabilityScore(**kwds),
        ]
    return metrics


def get_cfg(
    command,
    seed,
    model,
    distribution,
    data_kwds,
    results_path,
    output_shape=[],
    epochs=None,
    lr=None,
    distribution_kwds={},
    model_kwds={},
):
    name = "_".join((model, distribution))
    results_path = os.path.join(results_path, name)

    if command == "fit":
        compile_kwds = {
            "loss": get_loss(distribution, distribution_kwds),
            "optimizer": tf.keras.optimizers.Adam(lr=lr),
            # "run_eagerly": None,
        }
        fit_kwds = {
            "epochs": epochs,
            "validation_freq": 1,
            "callbacks": get_callbacks(lr, results_path),
        }
    elif command == "evaluate":
        compile_kwds = {
            "loss": NegativeLogarithmicLikelihood(
                get_distribution_class(distribution, distribution_kwds)
            ),
            "metrics": get_metrics(
                get_distribution_class(distribution, distribution_kwds),
                data_kwds["data_stats_path"],
            ),
        }
        fit_kwds = {}
    else:
        raise ValueError(f'Command "{command}" unknown')

    return Configuration(
        seed=seed,
        name=name,
        data_loader=load_data,
        data_loader_kwds=data_kwds,
        model_checkpoints=os.path.join(results_path, "mcp"),
        mlflow={
            "log_artifacts": results_path,
            "set_tags": {
                "model": model,
                "distribution": distribution,
            },
            "log_params": [
                {
                    "seed": seed,
                    "output_shape": output_shape,
                    "lr": lr,
                    "epochs": epochs,
                    "results_path": results_path,
                },
                {"data." + k: v for k, v in data_kwds.items()},
                {"distribution." + k: v for k, v in distribution_kwds.items()},
                {"model." + k: v for k, v in model_kwds.items() if len(str(v)) < 250},
            ],
        },
        model=get_model(
            architecture=model,
            history_size=data_kwds["history_size"],
            meta_columns=data_kwds["meta_columns"],
            prediction_size=data_kwds["prediction_size"],
            output_shape=output_shape,
            name=name,
        ),
        model_kwds=model_kwds,
        compile_kwds=compile_kwds,
        fit_kwds=fit_kwds,
    )
