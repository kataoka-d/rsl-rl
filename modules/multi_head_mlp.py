from functools import reduce

import torch
import torch.nn as nn

from rsl_rl.utils import get_param, resolve_nn_activation


class MultiHeadMLP(nn.Module):
    """Shared MLP encoder with heads for velocity, latent mean, and latent log variance."""

    def __init__(
        self,
        input_dim: int,
        v_dim: int | tuple[int, ...] | list[int],
        z_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str = "elu",
    ) -> None:
        super().__init__()

        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims must contain at least one layer.")

        activation_cls = resolve_nn_activation(activation)
        hidden_dims_processed = [input_dim if dim == -1 else dim for dim in hidden_dims]

        shared_layers: list[nn.Module] = []
        in_dim = input_dim
        for hidden_dim in hidden_dims_processed:
            shared_layers.append(nn.Linear(in_dim, hidden_dim))
            shared_layers.append(activation_cls)
            in_dim = hidden_dim
        self.shared_base = nn.Sequential(*shared_layers)

        last_hidden_dim = hidden_dims_processed[-1]

        if isinstance(v_dim, int):
            self.v_head = nn.Linear(last_hidden_dim, v_dim)
        else:
            total_v_out_dim = reduce(lambda x, y: x * y, v_dim)
            self.v_head = nn.Sequential(
                nn.Linear(last_hidden_dim, total_v_out_dim),
                nn.Unflatten(dim=-1, unflattened_size=v_dim),
            )

        self.mu_head = nn.Linear(last_hidden_dim, z_dim)
        self.log_var_head = nn.Linear(last_hidden_dim, z_dim)

    def init_weights(self, scales: float | tuple[float, ...]) -> None:
        """Orthogonally initialize all linear layers."""
        linear_index = 0
        for module in self.shared_base:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=get_param(scales, linear_index))
                nn.init.zeros_(module.bias)
                linear_index += 1

        heads = [self.v_head, self.mu_head, self.log_var_head]
        for head in heads:
            modules = head if isinstance(head, nn.Sequential) else [head]
            for module in modules:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=get_param(scales, -1))
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.shared_base(x)
        v_pred = self.v_head(features)
        mu = self.mu_head(features)
        log_var = self.log_var_head(features)
        return v_pred, mu, log_var
