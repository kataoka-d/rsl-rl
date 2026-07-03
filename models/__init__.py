# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Neural models for the learning algorithm."""

from .cnn_model import CNNModel
from .mlp_model import MLPModel
from .rnn_model import RNNModel
from .CustomMLPmodel import CustomMLPModel
from .AdaptiveLocomotionMLP_model import AdaptiveLocoMLPModel
from .DWL_rnn_model import DWLRNNModel
__all__ = [
    "CNNModel",
    "MLPModel",
    "RNNModel",
    "CustomMLPModel",
    "AdaptiveLocoMLPModel",
    "DWLRNNModel",
]
