# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState, MultiHeadMLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


class AdaptiveLocoMLPModel(nn.Module):
    """MLP-based neural model.

    This model uses a simple multi-layer perceptron (MLP) to process 1D observation groups. Observations can be
    normalized before being passed to the MLP. The output of the model can be either deterministic or
    stochastic, in which case a distribution module is used to sample the outputs.
    """

    is_recurrent: bool = False
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
        encoder_hidden_dims: tuple[int, ...] | list[int] = (768, 256, 64),
        v_dim: int = 3, #velocity estimate vの出力サイズ
        z_dim : int = 16, #encoder zの出力サイズ
        history_dim : int = 62 * 5,
        decoder_hidden_dims : tuple[int, ...] | list[int] = (512,512,128),
        decoder_output_dim: int | None = None,
        use_stochastic_latent: bool = False,
        clamp_log_var: tuple[float, float] = (-5.0, 2.0),
        
    ) -> None:
        """Initialize the MLP-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
        """
        super().__init__()

        self.v_dim = v_dim
        self.z_dim = z_dim
        self.history_dim = history_dim
        self.use_stochastic_latent = use_stochastic_latent
        self.clamp_log_var = clamp_log_var
        self.decoder_input_dim = self.v_dim + self.z_dim
        # Resolve observation groups and dimensions
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.history_normalizer = EmpiricalNormalization(self.history_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.history_normalizer = torch.nn.Identity()

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # MLP
        self.mlp = MLP(self._get_latent_dim(), mlp_output_dim, hidden_dims, activation)
        #encoder
        self.encoder = MultiHeadMLP(self._get_encoder_latent_dim(), v_dim, z_dim, encoder_hidden_dims, activation)
        #decoder
        self.decoder_output_dim = self.obs_dim if decoder_output_dim is None else decoder_output_dim
        self.decoder = MLP(self.decoder_input_dim, self.decoder_output_dim, decoder_hidden_dims, activation)

        # Initialize distribution-specific MLP weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the MLP model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MLP input latent
        latent = self.get_latent(obs, masks, hidden_state)
        # MLP forward pass
        mlp_output = self.mlp(latent)
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by concatenating and normalizing selected observation groups."""
        # Select and concatenate observations
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        current_obs = torch.cat(obs_list, dim=-1)
        current_obs_normalized = self.obs_normalizer(current_obs)

        obs_history = obs["actor_history"]
        if len(obs_history.shape) > 2:
            obs_history = obs_history.flatten(start_dim=1)
        obs_history = self.history_normalizer(obs_history)

        v_pred, mu, log_var = self.encoder(obs_history)
        log_var = torch.clamp(log_var, self.clamp_log_var[0], self.clamp_log_var[1])
        z_pred = self.reparameterize(mu, log_var) if self.use_stochastic_latent and self.training else mu

        self.last_v_pred = v_pred
        self.last_mu = mu
        self.last_log_var = log_var
        self.last_z = z_pred

        latent = torch.cat([current_obs_normalized, v_pred, z_pred], dim=-1)
        '''
        latent = torch.cat(obs_list, dim=-1)
        # Normalize observations
        latent = self.obs_normalizer(latent)
        '''
        return latent

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the internal state for recurrent models (no-op)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None`` for MLP)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach therecurrent hidden state for truncated backpropagation (no-op)."""
        pass

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """VAEの再パラメータ化トリック (Reparameterization Trick)"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    @property
    def predicted_velocity(self) -> torch.Tensor:
        """forwardで計算されたbase linkの推定速度を返す"""
        return self.last_v_pred
    
    @property
    def vae_params(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.last_mu, self.last_log_var

    @property
    def latent_sample(self) -> torch.Tensor:
        return self.last_z
    
    def decode_latent(self, z:torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two parameterizations of the distribution."""
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxMLPModel(self, verbose)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        if self.obs_normalization:
            # Select and concatenate observations
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            mlp_obs = torch.cat(obs_list, dim=-1)
            # Update the normalizer parameters
            self.obs_normalizer.update(mlp_obs)  # type: ignore
            obs_history = obs["actor_history"]
            if len(obs_history.shape) > 2:
                obs_history = obs_history.flatten(start_dim=1)
            self.history_normalizer.update(obs_history)  # type: ignore

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        """the MLP input size = obs history + v,z(encoded obs history)"""
        return self.obs_dim + self.v_dim + self.z_dim
    
    def _get_encoder_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the encoder MLP head"""
        return self.history_dim


class _TorchMLPModel(nn.Module):
    """Exportable MLP model for JIT."""

    def __init__(self, model: AdaptiveLocoMLPModel) -> None:
        """Create a TorchScript-friendly copy of an MLPModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_normalizer = copy.deepcopy(model.history_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, current_obs: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference from current observation and proprioceptive history."""
        current_obs = self.obs_normalizer(current_obs)
        if obs_history.dim() > 2:
            obs_history = obs_history.flatten(start_dim=1)
        obs_history = self.history_normalizer(obs_history)
        v_pred, mu, _ = self.encoder(obs_history)
        latent = torch.cat([current_obs, v_pred, mu], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for MLP exports)."""
        pass


class _OnnxMLPModel(nn.Module):
    """Exportable MLP model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: AdaptiveLocoMLPModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an MLPModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_normalizer = copy.deepcopy(model.history_normalizer)
        self.encoder = copy.deepcopy(model.encoder)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim
        self.history_size = model.history_dim

    def forward(self, current_obs: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        current_obs = self.obs_normalizer(current_obs)
        if obs_history.dim() > 2:
            obs_history = obs_history.flatten(start_dim=1)
        obs_history = self.history_normalizer(obs_history)
        v_pred, mu, _ = self.encoder(obs_history)
        latent = torch.cat([current_obs, v_pred, mu], dim=-1)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size), torch.zeros(1, self.history_size))

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", "actor_history"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
