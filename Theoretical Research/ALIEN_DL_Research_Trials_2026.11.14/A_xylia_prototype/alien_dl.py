"""ALIEN-DL: a small research prototype of resource-coupled synaptic computation.

This is an experimental architecture, not a biologically faithful simulator.
It implements three independently switchable couplings:
  R: local allocation controls short-term resource recovery;
  P: local allocation gates synaptic parameter updates;
  A: local allocation drives structural persistence/pruning.

The implementation is intentionally compact so that the mechanisms can be
inspected and ablated. Structural births and multi-resource biochemical pools
are not yet implemented.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class AlienConfig:
    n_compartments: int = 8
    micro_steps: int = 3
    rho: float = 0.55
    alpha_depression: float = 0.18
    alpha_facilitation: float = 0.10
    tau_u: float = 3.0
    u_rest: float = 0.25
    allocation_lr: float = 0.025
    theta_lr: float = 0.025
    structural_rate: float = 0.08
    prune_threshold: float = 0.15
    r_coupling: bool = True
    p_coupling: bool = True
    a_coupling: bool = True


class AlienNet(nn.Module):
    """Dendritic compartments fed by stateful, resource-coupled synapses."""

    def __init__(self, n_features: int, n_classes: int, config: Optional[AlienConfig] = None):
        super().__init__()
        self.config = config or AlienConfig()
        c, f = self.config.n_compartments, n_features
        self.n_features = f
        self.n_compartments = c

        # Baseline synaptic efficacy. A positive/negative unconstrained efficacy
        # is allowed; local allocation is represented separately.
        self.theta = nn.Parameter(torch.empty(c, f))
        nn.init.kaiming_uniform_(self.theta, a=5 ** 0.5)

        # Each compartment has a simplex of resource shares over its afferents.
        self.alloc_logits = nn.Parameter(torch.zeros(c, f))

        # Structural persistence logits. Initially, all candidate synapses exist.
        self.zeta = nn.Parameter(torch.full((c, f), 2.0), requires_grad=False)
        self.register_buffer("structural_mask", torch.ones(c, f))

        self.compartment_bias = nn.Parameter(torch.zeros(c))
        self.readout = nn.Linear(c, n_classes)
        self.register_buffer("resource_budget", torch.ones(c))
        self._last_pi: Optional[torch.Tensor] = None
        self._last_q: Optional[torch.Tensor] = None

    def allocation(self) -> torch.Tensor:
        """Return active-synapse shares with mean 1 per compartment.

        The underlying probability mass is normalized over extant synapses only;
        pruning therefore redistributes the fixed abstract budget to survivors.
        """
        active = self.structural_mask > 0
        masked_logits = self.alloc_logits.masked_fill(~active, -1.0e9)
        probs = torch.softmax(masked_logits, dim=-1)
        n_active = active.sum(dim=-1, keepdim=True).clamp_min(1).to(probs.dtype)
        return probs * n_active

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Inputs are standardized by the benchmark pipeline. tanh bounds the
        # presynaptic drive, making the internal resource update numerically safe.
        x_drive = torch.tanh(x)
        batch = x.shape[0]
        pi = self.allocation()
        self._last_pi = pi

        if self.config.r_coupling:
            recovery_share = pi
        else:
            recovery_share = torch.ones_like(pi)

        # Per-example, per-synapse short-term state. These are transient states,
        # reset for each independent tabular example in this first prototype.
        r = torch.ones(batch, self.n_compartments, self.n_features, device=x.device, dtype=x.dtype)
        u = torch.full_like(r, self.config.u_rest)
        activity = x_drive.abs().unsqueeze(1)

        for _ in range(self.config.micro_steps):
            dr = (self.config.rho * recovery_share.unsqueeze(0) * (1.0 - r)
                  - self.config.alpha_depression * u * r * activity)
            du = ((self.config.u_rest - u) / self.config.tau_u
                  + self.config.alpha_facilitation * (1.0 - u) * activity)
            r = torch.clamp(r + dr, 0.0, 1.0)
            u = torch.clamp(u + du, 0.0, 1.0)

        if self.config.a_coupling:
            structural = torch.sigmoid(self.zeta) * self.structural_mask
        else:
            structural = self.structural_mask

        q = (self.theta.unsqueeze(0)
             * u * r
             * structural.unsqueeze(0)
             * x_drive.unsqueeze(1))
        self._last_q = q
        compartment = torch.tanh(q.sum(dim=-1) + self.compartment_bias)
        return self.readout(compartment)

    @torch.no_grad()
    def allocation_step(self, utility: torch.Tensor) -> None:
        """Multiplicative-weights allocation update on each active simplex.

        Utility is a synapse contribution score, -q * dL/dq, averaged over
        examples. It is a heuristic selection signal, not claimed to be the
        exact gradient of loss with respect to the resource budget.
        """
        active = self.structural_mask > 0
        probs = torch.softmax(self.alloc_logits.masked_fill(~active, -1.0e9), dim=-1)
        utility = torch.nan_to_num(utility, nan=0.0, posinf=0.0, neginf=0.0)
        utility = utility.masked_fill(~active, 0.0)
        # The raw contribution scores can be tiny on normalized small datasets.
        # Normalize their row-wise scale so the allocation timescale is not
        # accidentally determined by the units of the loss or input features.
        scale = utility.abs().sum(dim=-1, keepdim=True) / active.sum(dim=-1, keepdim=True).clamp_min(1)
        utility = utility / scale.clamp_min(1e-8)
        centered = utility - (probs * utility).sum(dim=-1, keepdim=True)
        updated = self.alloc_logits + self.config.allocation_lr * centered
        updated = updated - updated.masked_fill(~active, 0.0).sum(dim=-1, keepdim=True) / active.sum(dim=-1, keepdim=True).clamp_min(1)
        self.alloc_logits.copy_(torch.where(active, updated.clamp(-8.0, 8.0), self.alloc_logits))

    @torch.no_grad()
    def synaptic_parameter_step(self, grad_theta: torch.Tensor, lr_scale: float = 1.0) -> None:
        """SGD update with optional allocation-gated local learning rate."""
        if self.config.p_coupling:
            multiplier = self.allocation().detach()
        else:
            multiplier = torch.ones_like(self.theta)
        self.theta.add_(grad_theta * multiplier, alpha=-self.config.theta_lr * lr_scale)

    @torch.no_grad()
    def structural_epoch_step(self) -> int:
        """Update persistence from relative allocation and prune at epoch boundary.

        The allocation budget is redistributed by the softmax over currently
        active candidates. This prototype supports death/pruning, not birth.
        """
        if not self.config.a_coupling:
            return 0
        pi = self.allocation()
        active = self.structural_mask > 0
        self.zeta.add_(self.config.structural_rate * (pi - 1.0) * active)
        self.zeta.clamp_(-8.0, 8.0)
        proposed = (torch.sigmoid(self.zeta) >= self.config.prune_threshold).to(self.structural_mask.dtype)
        proposed = proposed * self.structural_mask
        # Keep at least one incoming synapse per compartment.
        for k in range(self.n_compartments):
            if proposed[k].sum() < 1:
                best = torch.argmax(self.zeta[k])
                proposed[k, best] = 1.0
        removed = int((self.structural_mask - proposed).clamp_min(0).sum().item())
        self.structural_mask.copy_(proposed)
        return removed

    def diagnostics(self) -> dict[str, float]:
        with torch.no_grad():
            pi = self.allocation()
            probs = pi / self.n_features
            entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
            effective = entropy.exp().mean().item()
            n_active = self.structural_mask.sum().item()
            return {
                "allocation_gini": float(_mean_row_gini(pi).item()),
                "effective_fanin": float(effective),
                "active_synapses": float(n_active),
                "total_synapses": float(self.n_compartments * self.n_features),
                "active_fraction": float(n_active / (self.n_compartments * self.n_features)),
            }


def _mean_row_gini(x: torch.Tensor) -> torch.Tensor:
    # Gini over nonnegative resource shares, averaged across compartments.
    sorted_x, _ = torch.sort(x.clamp_min(0), dim=-1)
    n = x.shape[-1]
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    numerator = (2 * idx - n - 1) * sorted_x
    denom = n * sorted_x.sum(dim=-1).clamp_min(1e-12)
    return (numerator.sum(dim=-1) / denom).mean()


class MLP(nn.Module):
    """Conventional two-layer MLP baseline."""
    def __init__(self, n_features: int, n_classes: int, width: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, width),
            nn.Tanh(),
            nn.Linear(width, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_matched_width(n_features: int, n_classes: int, target_params: int) -> int:
    """Find the nearest hidden width for a two-layer MLP parameter budget."""
    best_width, best_diff = 1, float("inf")
    for width in range(1, 1025):
        count = n_features * width + width + width * n_classes + n_classes
        diff = abs(count - target_params)
        if diff < best_diff:
            best_width, best_diff = width, diff
    return best_width
