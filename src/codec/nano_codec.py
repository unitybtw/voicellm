"""Native residual vector quantization; footprint and bitrate depend on config."""

import math
from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """Single-stage vector quantizer with straight-through estimator."""
    def __init__(self, codebook_size: int = 1024, latent_dim: int = 64):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        # Initialize codebook with uniform distribution
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, D, T] continuous latent representation
        Returns:
            z_q: [B, D, T] quantized latent
            indices: [B, T] discrete token indices
            loss: commitment loss
        """
        # [B, D, T] -> [B, T, D]
        z_perm = z.permute(0, 2, 1).contiguous()
        flat_z = z_perm.view(-1, self.latent_dim)

        # Calculate distances: ||z - e||^2 = z^2 - 2*z*e + e^2
        w = self.embedding.weight
        distances = (
            torch.sum(flat_z ** 2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_z, w.t())
            + torch.sum(w ** 2, dim=1, keepdim=True).t()
        )

        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view(z_perm.shape)

        # Commitment & codebook loss
        loss = F.mse_loss(z_q, z_perm.detach()) + 0.25 * F.mse_loss(z_q.detach(), z_perm)

        # Straight-through gradient estimator
        z_q = z_perm + (z_q - z_perm).detach()

        # [B, T, D] -> [B, D, T]
        z_q = z_q.permute(0, 2, 1).contiguous()
        indices = indices.view(z.shape[0], z.shape[2])

        return z_q, indices, loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decodes indices [B, T] back to continuous vectors [B, D, T]."""
        z_q = self.embedding(indices) # [B, T, D]
        return z_q.permute(0, 2, 1).contiguous()


class ResidualVectorQuantizer(nn.Module):
    """Hierarchical / Residual Vector Quantizer (RVQ) with N stages."""
    def __init__(self, num_quantizers: int = 4, codebook_size: int = 1024, latent_dim: int = 64):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.quantizers = nn.ModuleList([
            VectorQuantizer(codebook_size, latent_dim) for _ in range(num_quantizers)
        ])

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [B, D, T]
        Returns:
            quantized: [B, D, T]
            indices: [B, num_quantizers, T]
            total_loss: scalar loss
        """
        residual = z
        quantized = 0.0
        indices_list = []
        total_loss = 0.0

        for q in self.quantizers:
            z_q, idx, loss = q(residual)
            residual = residual - z_q.detach()
            quantized = quantized + z_q.detach()
            indices_list.append(idx)
            total_loss = total_loss + loss

        indices = torch.stack(indices_list, dim=1) # [B, K, T]
        quantized = z + (quantized - z).detach()
        return quantized, indices, total_loss

    def decode_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decodes [B, K, T] indices back to continuous latent [B, D, T]."""
        quantized = 0.0
        for i, q in enumerate(self.quantizers):
            quantized = quantized + q.decode_indices(indices[:, i, :])
        return quantized
