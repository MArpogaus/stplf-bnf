# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : visualization.py
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
import matplotlib
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np

sns.set(style="ticks", context="paper")


# function definitions ########################################################
def plot_patches(ds,
                 x_vdim,
                 x_hdim,
                 y_vdim,
                 y_hdim,
                 N=3,
                 historic_columns=[],
                 horizon_columns=[],
                 prediction_columns=[],
                 title_map={},
                 y_label_map={},
                 xy_ch_connect=None,
                 gridspec_kw={},
                 fig_kw={},
                 heatmap_kw={}):
    x, y = next(ds.take(1).as_numpy_iterator())
    x_patches = x[:N]
    y_patches = y[:N]

    fig = plt.figure(**fig_kw)
    gs = fig.add_gridspec(2, N, **gridspec_kw)

    last_y_ax = {}

    y_label_kw = dict(rotation=0,
                      labelpad=20,
                      verticalalignment='center_baseline',
                      horizontalalignment='right')

    x_vmin = x_patches[:N].min(axis=1).min(axis=0)
    x_vmax = x_patches[:N].max(axis=1).max(axis=0)

    y_off = np.where(len(horizon_columns), y_vdim, 0)
    x_off = np.where(len(horizon_columns), x_vdim, 0)

    x_columns = sorted(set(historic_columns + horizon_columns))
    y_columns = sorted(prediction_columns)

    x_column_ch = {k: c for c, k in enumerate(x_columns)}
    y_column_ch = {k: c for c, k in enumerate(y_columns)}

    for n in range(N):
        x = x_patches[n]
        x_ch = x.shape[-1]

        x = x.reshape(-1, x_hdim, x_ch)

        sgs = gs[0, n].subgridspec(nrows=x_ch,
                                   ncols=1,
                                   hspace=0.1,
                                   wspace=0.1)

        for col in x_columns:
            c = x_column_ch[col]
            ax = fig.add_subplot(sgs[c])
            sns.heatmap(x[:, :, c],
                        ax=ax,
                        linewidth=0.2,
                        xticklabels=c == 0,
                        yticklabels=n == 0,
                        cbar=False,
                        vmin=x_vmin[c],
                        vmax=x_vmax[c],
                        **heatmap_kw.get('x', {}).get(col, {}))
            if n == 0:
                ax.set_ylabel(y_label_map.get('x', {}).get(col, col),
                              **y_label_kw)
            if c == 0:
                ax.tick_params(top=True, bottom=False,
                               labeltop=True, labelbottom=False)
                ax.set_title(title_map.get('x', 'x'))
            if n > 0 and xy_ch_connect is not None and col == xy_ch_connect[1][0]:
                patch = matplotlib.patches.ConnectionPatch(
                    xyA=(1, 0.5 / y_vdim),
                    xyB=(0.0, ((x_vdim + y_off) - xy_ch_connect[1][1] - 0.5) /
                         (x_vdim + y_off)),
                    coordsA="axes fraction",
                    coordsB="axes fraction",
                    axesA=last_y_ax[xy_ch_connect[0][0]],
                    axesB=ax,
                    arrowstyle='->',
                    connectionstyle="arc,angleA=0,armA=7,angleB=-180,armB=7,rad=3",
                    color='k'
                )
                ax.add_artist(patch)
            if n == (N - 1) and col in historic_columns:
                ax.annotate('History',
                            xy=(1, (0.5 * x_vdim + y_off) / (x_vdim + y_off)),
                            xytext=(1.05, (0.5 * x_vdim + y_off) /
                                    (x_vdim + y_off)),
                            xycoords='axes fraction',
                            ha='left', va='center',
                            bbox=dict(boxstyle='square', fc='white'),
                            arrowprops=dict(
                                arrowstyle=f'-[, widthB={x_vdim/2 - 0.2}, lengthB=0.2',
                                color='k'))
            if n == (N - 1) and col in horizon_columns:
                ax.annotate('Horizon',
                            xy=(1, (0.5 * y_vdim) / (x_off + y_vdim)),
                            xytext=(1.05, (0.5 * y_vdim) / (x_off + y_vdim)),
                            xycoords='axes fraction',
                            ha='left', va='center',
                            bbox=dict(boxstyle='square', fc='white'),
                            arrowprops=dict(
                                arrowstyle=f'-[, widthB={y_vdim/2 - 0.2}, lengthB=0.2',
                                color='k'))

        y_ch = y_patches.shape[2]
        y = y_patches[n].reshape(y_vdim, y_hdim, y_ch)

        sgs = gs[1, n].subgridspec(nrows=y_ch,
                                   ncols=1,
                                   # hspace=0.1,
                                   wspace=0.1)
        for col in y_columns:
            c = y_column_ch[col]
            ax = fig.add_subplot(sgs[c])
            sns.heatmap(y[:, :, c],
                        ax=ax,
                        linewidth=0.2,
                        xticklabels=False,
                        yticklabels=n == 0,
                        cbar=False,
                        vmin=x_vmin[x_column_ch[col]],
                        vmax=x_vmax[x_column_ch[col]],
                        **heatmap_kw.get('y', {}).get(col, {}))
            if n == 0:
                ax.set_ylabel(y_label_map.get('y', {}).get(col, col),
                              **y_label_kw)
            if c == 0:
                ax.set_title(title_map.get('y', 'y'))

            last_y_ax.update({col: ax})

    return fig


def plot_forecast(model,
                  x,
                  y,
                  history_size,
                  horizon_size,
                  historic_columns,
                  horizon_columns,
                  prediction_columns,
                  sd=[1, 2],
                  horizon=1,
                  shift=0,
                  fig_kw={}):

    columns = sorted(
        set(historic_columns + horizon_columns + prediction_columns))
    x_columns = sorted(set(historic_columns + horizon_columns))
    # y_columns = sorted(prediction_columns)

    x_column_ch = {k: c for c, k in enumerate(x_columns)}
    # y_column_ch = {k: c for c, k in enumerate(y_columns)}

    N = shift + horizon
    fig, ax = plt.subplots(2, **fig_kw)

    t = np.array(range(history_size + (N - shift) * horizon_size))
    t_hori = np.array(range((N - shift) * horizon_size)) + history_size

    for k in columns:
        if k in historic_columns and k in horizon_columns:
            hist = x[shift, :history_size, x_column_ch[k]].flatten()
            hori = x[shift:N, history_size:, x_column_ch[k]].flatten()
            dat = np.concatenate([hist, hori]).flatten()
            print(k, y.shape, t.shape)
            ax_idx = 0
        elif k in historic_columns and k in prediction_columns:
            hist = x[shift, :history_size, x_column_ch[k]].flatten()
            hori = y[shift:N].flatten()
            dat = np.concatenate([hist, hori]).flatten()
            print(k, y.shape, t.shape)
            ax_idx = 1
        ax[ax_idx].plot(t, dat, label=k)

    pred_mu = []
    pred_log_sigma = []

    for n in range(shift, N):

        input_data = x[n].reshape(1, -1, 3)
        pred = model.predict(input_data).reshape(48, 2)
        pred_mu.append(pred[:, 0])
        pred_log_sigma.append(pred[:, 1])

    pred_mu = np.stack(pred_mu).flatten()
    pred_log_sigma = np.stack(pred_log_sigma).flatten()

    ax[1].plot(t_hori, pred_mu, label='mean of prediction',
               c="black", linewidth=2)

    for sd in sd:
        pred_sd_p = pred_mu + sd * (np.exp(pred_log_sigma))
        pred_sd_m = pred_mu - sd * (np.exp(pred_log_sigma))

        ax[1].plot(t_hori, pred_sd_p, 'b', linewidth=0.5)
        ax[1].plot(t_hori, pred_sd_m, 'b', linewidth=0.5)

        ax[1].fill(np.concatenate([t_hori, t_hori[::-1]]),
                   np.concatenate([pred_sd_p,
                                   pred_sd_m[::-1]]),
                   alpha=1 / (2 * sd),
                   fc='lightskyblue')

    ax[0].legend()
    ax[1].legend()

    plt.show()
