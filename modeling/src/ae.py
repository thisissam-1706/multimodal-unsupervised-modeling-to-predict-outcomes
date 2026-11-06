"""Masked denoising autoencoder with a 4-class head (task B4).

Input per row: x (preprocessed values, D columns) and mask (D columns, 1 = observed).
The encoder sees [x * mask, mask], so the value stored in a missing cell never matters.

    encoder: 2D -> 256 -> 128 -> 64 (latent Z)
    decoder: 64 -> 128 -> 256 -> D   (reconstructs the preprocessed inputs)
    head:    64 -> 32 -> 4 logits
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU}


def _mlp(sizes: list[int], activation: str, dropout: float, last_activation: bool) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2 or last_activation:
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class MaskedDAE(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...] = (256, 128), latent: int = 64,
                 head_hidden: int = 32, n_classes: int = 4, dropout: float = 0.1,
                 activation: str = "relu"):
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {sorted(ACTIVATIONS)}")
        self.n_features = n_features
        self.latent_dim = latent
        enc = [2 * n_features, *hidden, latent]
        self.encoder = _mlp(enc, activation, dropout, last_activation=False)
        self.decoder = _mlp(enc[::-1][:-1] + [n_features], activation, dropout, last_activation=False)
        self.head = _mlp([latent, head_hidden, n_classes], activation, dropout, last_activation=False)

    @staticmethod
    def _mask(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        return torch.ones_like(x) if mask is None else mask

    def encode(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Latent Z, shape (N, latent). mask=None means every cell is observed."""
        m = self._mask(x, mask)
        return self.encoder(torch.cat([x * m, m], dim=1))

    def reconstruct(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Reconstruction of all D input columns, shape (N, D)."""
        return self.decoder(self.encode(x, mask))

    def classify(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Class logits in contract order, shape (N, n_classes)."""
        return self.head(self.encode(x, mask))

    def forward(self, x, mask=None):
        z = self.encode(x, mask)
        return self.decoder(z), self.head(z)


def masked_reconstruction_loss(recon: torch.Tensor, target: torch.Tensor,
                               observed: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    """Huber loss averaged over observed cells only (observed == 1). Missing cells contribute 0."""
    per_cell = F.huber_loss(recon, torch.where(observed > 0, target, recon.detach()),
                            reduction="none", delta=delta)
    return (per_cell * observed).sum() / observed.sum().clamp(min=1.0)


def corrupt(x: torch.Tensor, mask: torch.Tensor, groups: torch.Tensor, rate: float,
            generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Hide a fraction `rate` of the OBSERVED features (whole one-hot blocks for categoricals).

    groups[j] = feature index of output column j. Hidden cells get value 0 and mask 0.
    Returns (x_corrupted, mask_corrupted). Already-missing cells are left as they are.
    """
    n_groups = int(groups.max().item()) + 1
    hide_feature = torch.rand((x.shape[0], n_groups), generator=generator) < rate
    hide = hide_feature[:, groups] & (mask > 0)
    keep = (~hide).to(x.dtype)
    return x * keep, mask * keep
