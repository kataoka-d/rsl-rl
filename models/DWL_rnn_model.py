# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import RNN, HiddenState
from rsl_rl.utils import unpad_trajectories


def _resolve_activation(name: str) -> nn.Module:
    """Create an activation module from a small, dependency-free name map."""
    activations = {
        "elu": nn.ELU,
        "relu": nn.ReLU,
        "selu": nn.SELU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
    }
    key = name.lower()
    if key not in activations:
        raise ValueError(f"Unsupported activation for denoising decoder: {name}")
    return activations[key]()


def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...] | list[int], output_dim: int, activation: str) -> nn.Sequential:
    """Build a plain MLP used by the denoising decoder."""
    layers: list[nn.Module] = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(last_dim, hidden_dim))
        layers.append(_resolve_activation(activation))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class DWLRNNModel(MLPModel):
    """Denoising World Model Learning RNN policy model.

    This model uses an RNN encoder for the actor latent and can train a decoder that reconstructs
    privileged/full state targets from that latent.
    """

    is_recurrent: bool = True
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        denoising_cfg: dict | None = None,
    ) -> None:
        """Initialize the RNN-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution.
            rnn_type: Type of RNN to use ("lstm" or "gru").
            rnn_hidden_dim: Dimension of the RNN hidden state.
            rnn_num_layers: Number of RNN layers.
            denoising_cfg: Optional decoder configuration for Denoising World Model Learning.
        """
        self.latent_dim = rnn_hidden_dim
        self.denoising_cfg = denoising_cfg

        # Initialize the parent MLP model
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )

        # RNN
        self.rnn = RNN(self.obs_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)
        self._last_latent: torch.Tensor | None = None

        # Optional DWL decoder: reconstruct privileged/full state from the actor latent.
        self.denoising_decoder: nn.Module | None = None
        self.denoising_target_groups: list[str] = []
        self.denoising_l1_coef = 0.002
        self.denoising_reconstruction_loss = "l2"
        if denoising_cfg is not None:
            target_groups = denoising_cfg.get("target_groups")
            target_obs_set = denoising_cfg.get("target_obs_set")
            if target_groups is None and target_obs_set is not None:
                target_groups = obs_groups[target_obs_set]
            if isinstance(target_groups, str):
                target_groups = [target_groups]
            if not target_groups:
                raise ValueError("denoising_cfg requires 'target_groups' or 'target_obs_set'.")

            self.denoising_target_groups = list(target_groups)
            self.denoising_l1_coef = float(denoising_cfg.get("l1_coef", self.denoising_l1_coef))
            self.denoising_reconstruction_loss = denoising_cfg.get("reconstruction_loss", "l2").lower()
            target_dim = self._get_obs_groups_dim(obs, self.denoising_target_groups)
            decoder_hidden_dims = denoising_cfg.get("decoder_hidden_dims", hidden_dims)
            decoder_activation = denoising_cfg.get("activation", activation)
            self.denoising_decoder = _build_mlp(
                rnn_hidden_dim,
                decoder_hidden_dims,
                target_dim,
                decoder_activation,
            )

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by passing normalized observation groups through the RNN."""
        # Extract and concatenate observation groups and normalize
        latent = super().get_latent(obs)
        # Pass through the RNN
        latent = self.rnn(latent, masks, hidden_state)
        if latent.dim() >= 3 and latent.shape[0] == 1:
            latent = latent.squeeze(0)
        self._last_latent = latent
        return latent

    def compute_denoising_loss(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        latent: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Compute DWL reconstruction loss from the actor latent to privileged/full state targets."""
        if self.denoising_decoder is None:
            return None
        if latent is None:
            latent = self._last_latent
        if latent is None:
            raise RuntimeError("Denoising loss requested before the actor latent was computed.")

        prediction = self.denoising_decoder(latent)
        target = self._get_denoising_target(obs, masks, prediction)
        if self.denoising_reconstruction_loss == "mse":
            reconstruction_loss = (prediction - target).pow(2).mean()
        elif self.denoising_reconstruction_loss == "l2":
            reconstruction_loss = torch.linalg.vector_norm(prediction - target, ord=2, dim=-1).mean()
        else:
            raise ValueError(f"Unsupported denoising reconstruction loss: {self.denoising_reconstruction_loss}")

        latent_l1 = latent.abs().mean()
        denoising_loss = reconstruction_loss + self.denoising_l1_coef * latent_l1
        return {
            "loss": denoising_loss,
            "reconstruction": reconstruction_loss,
            "latent_l1": latent_l1,
        }

    def _get_denoising_target(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        """Extract and shape target observation groups to match the decoder output."""
        batch_shape = tuple(obs.batch_size)
        targets = []
        for group in self.denoising_target_groups:
            target = obs[group]
            target = target.reshape(*batch_shape, -1)
            targets.append(target)
        target = torch.cat(targets, dim=-1)

        if masks is not None and target.dim() == masks.dim() + 1:
            target = unpad_trajectories(target, masks)
        if target.shape[:-1] != prediction.shape[:-1]:
            target = target.reshape(-1, target.shape[-1])
        return target.to(device=prediction.device, dtype=prediction.dtype)

    @staticmethod
    def _get_obs_groups_dim(obs: TensorDict, groups: list[str]) -> int:
        """Return the flattened feature size for a list of TensorDict observation groups."""
        dim = 0
        batch_ndim = len(obs.batch_size)
        for group in groups:
            if group not in obs.keys():
                raise KeyError(f"Denoising target group '{group}' was not found in observations.")
            shape = obs[group].shape[batch_ndim:]
            group_dim = 1
            for size in shape:
                group_dim *= size
            dim += group_dim
        return dim

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the recurrent hidden state of the RNN."""
        self.rnn.reset(dones, hidden_state)

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state of the RNN."""
        return self.rnn.hidden_state  # type: ignore

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach the recurrent hidden state for truncated backpropagation."""
        self.rnn.detach_hidden_state(dones)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        if isinstance(self.rnn.rnn, nn.LSTM):
            return _TorchLSTMModel(self)
        elif isinstance(self.rnn.rnn, nn.GRU):
            return _TorchGRUModel(self)
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn.rnn)}")

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxRNNModel(self, verbose)

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.latent_dim


class _TorchGRUModel(nn.Module):
    """Exportable GRU model for JIT."""

    def __init__(self, model: DWLRNNModel) -> None:
        """Create a TorchScript-friendly copy of a GRU-based DWLRNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.rnn.cpu()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one GRU inference step and update hidden states."""
        x = self.obs_normalizer(x)
        x, h = self.rnn(x.unsqueeze(0), self.hidden_state)
        self.hidden_state[:] = h  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported GRU hidden states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore


class _TorchLSTMModel(nn.Module):
    """Exportable LSTM model for JIT."""

    def __init__(self, model: DWLRNNModel) -> None:
        """Create a TorchScript-friendly copy of an LSTM-based DWLRNNModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
        self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one LSTM inference step and update hidden and cell states."""
        x = self.obs_normalizer(x)
        x, (h, c) = self.rnn(x.unsqueeze(0), (self.hidden_state, self.cell_state))
        self.hidden_state[:] = h  # type: ignore
        self.cell_state[:] = c  # type: ignore
        x = x.squeeze(0)
        out = self.mlp(x)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset exported LSTM hidden and cell states to zeros."""
        self.hidden_state[:] = 0.0  # type: ignore
        self.cell_state[:] = 0.0  # type: ignore


class _OnnxRNNModel(nn.Module):
    """Exportable RNN model for ONNX."""

    is_recurrent: bool = True

    def __init__(self, model: DWLRNNModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around a DWLRNNModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.rnn = copy.deepcopy(model.rnn.rnn)  # Access underlying torch module to avoid wrapper logic during export
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        # Detect RNN type
        if isinstance(self.rnn, nn.LSTM):
            self.rnn_type = "lstm"
        elif isinstance(self.rnn, nn.GRU):
            self.rnn_type = "gru"
        else:
            raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

        self.input_size = model.obs_dim
        self.hidden_size = self.rnn.hidden_size
        self.num_layers = self.rnn.num_layers

    def forward(
        self, obs: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(obs)

        if self.rnn_type == "lstm":
            x, (h, c) = self.rnn(x.unsqueeze(0), (h_in, c_in))
            x = x.squeeze(0)
            out = self.mlp(x)
            out = self.deterministic_output(out)
            return out, h, c
        else:
            x, h = self.rnn(x.unsqueeze(0), h_in)
            x = x.squeeze(0)
            out = self.mlp(x)
            out = self.deterministic_output(out)
            return out, h, None

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        obs = torch.zeros(1, self.input_size)
        h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
        if self.rnn_type == "lstm":
            c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
            return (obs, h_in, c_in)
        return (obs, h_in)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        if self.rnn_type == "lstm":
            return ["obs", "h_in", "c_in"]
        return ["obs", "h_in"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        if self.rnn_type == "lstm":
            return ["actions", "h_out", "c_out"]
        return ["actions", "h_out"]
