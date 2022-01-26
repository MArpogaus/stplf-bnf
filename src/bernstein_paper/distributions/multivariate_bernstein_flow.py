# -*- time-stamp-pattern: "changed[\s]+:[\s]+%%$"; -*-
# AUTHOR INFORMATION ##########################################################
# file    : multivariate_bernstein_flow.py
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

from tensorflow_probability import distributions as tfd

from bernstein_flow.distributions import BernsteinFlow


class MultivariateBernsteinFlow(tfd.Blockwise):
    """
    This class implements a `tfd.TransformedDistribution` using Bernstein
    polynomials as the bijector.
    """

    def __init__(self,
                 pvector: tf.Tensor,
                 distribution: tfd.Distribution = tfd.Normal(loc=0., scale=1.)
                 ) -> tfd.Distribution:
        """
        Generate the flow for the given parameter vector. This would be
        typically the output of a neural network.

        To use it as a loss function see
        `bernstein_flow.losses.BernsteinFlowLoss`.

        :param      pvector:       The paramter vector.
        :type       pvector:       Tensor
        :param      distribution:  The base distribution to use.
        :type       distribution:  Distribution

        :returns:   The transformed distribution (normalizing flow)
        :rtype:     Distribution
        """
        num_dist = pvector.shape[1]

        flows = []
        for d in range(num_dist):
            flow = BernsteinFlow(pvector[:, d])
            flows.append(flow)

        joint = tfd.JointDistributionSequential(flows, name='joint_bs_flows')
        super().__init__(flows, name='MultivariateBernsteinFlow')

    def _stddev(self):
        return self._flatten_and_concat_event(self._distribution.stddev())

    def _quantile(self, value):
        qs = [d._quantile(value) for d in self.distributions]
        return self._flatten_and_concat_event(qs)

    def _cdf(self, value):
        qs = [d._cdf(value) for d in self.distributions]
        return self._flatten_and_concat_event(qs)
