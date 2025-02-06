import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class EMACodebook(nn.Module):
    def __init__(
        self,
        n_codes,
        embedding_dim,
        no_random_restart=True,
        restart_thres=1.0,
        ema_decay=0.99,
        eps=1e-5,  # To avoid division by zero
    ):
        super().__init__()
        self.register_buffer("embeddings", torch.randn(n_codes, embedding_dim))
        self.register_buffer("N", torch.zeros(n_codes))
        self.register_buffer("z_avg", self.embeddings.data.clone())

        self.n_codes = n_codes
        self.embedding_dim = embedding_dim
        self._need_init = True
        self.no_random_restart = no_random_restart
        self.restart_thres = restart_thres
        self.freeze_codebook = False
        self.ema_decay = ema_decay
        self.eps = eps

    def reset_parameters(self):
        """Reset parameters if necessary."""
        pass

    def _tile(self, x):
        """Tile input tensor to match the required number of codes."""
        d, ew = x.shape
        if d < self.n_codes:
            n_repeats = (self.n_codes + d - 1) // d
            std = 0.01 / np.sqrt(ew)
            x = x.repeat(n_repeats, 1)
            x = x + torch.randn_like(x) * std
        return x

    def _init_embeddings(self, z):
        """Initialize codebook with encoder outputs."""
        self._need_init = False
        flat_inputs = z.view(-1, self.embedding_dim)
        y = self._tile(flat_inputs)

        _k_rand = y[torch.randperm(y.shape[0])][: self.n_codes]
        if dist.is_initialized():
            dist.broadcast(_k_rand, 0)
        self.embeddings.data.copy_(_k_rand)
        self.z_avg.data.copy_(_k_rand)
        self.N.data.copy_(torch.ones(self.n_codes))

    def forward(self, z):
        """Forward pass with quantization and EMA update."""
        if self._need_init and self.training and not self.freeze_codebook:
            self._init_embeddings(z)

        flat_inputs = z.view(-1, self.embedding_dim)
        distances = (
            (flat_inputs**2).sum(dim=1, keepdim=True)
            - 2 * flat_inputs @ self.embeddings.t()
            + (self.embeddings.t() ** 2).sum(dim=0, keepdim=True)
        )  # [bt, c]

        encoding_indices = torch.argmin(distances, dim=1)
        encoding_indices = encoding_indices.view(*z.shape[:2])  # [b, t]

        embeddings = F.embedding(encoding_indices, self.embeddings)  # [b, t, c]

        commitment_loss = 0.25 * F.mse_loss(z, embeddings.detach())

        # EMA codebook update
        if self.training and not self.freeze_codebook:
            encodings_one_hot = F.one_hot(encoding_indices, num_classes=self.n_codes).float()
            encodings_one_hot = encodings_one_hot.view(-1, self.n_codes)

            # Update the usage count for each code
            code_usage = encodings_one_hot.sum(dim=0)
            self.N.data.mul_(self.ema_decay).add_(code_usage, alpha=1 - self.ema_decay)

            # Compute the new centroid for each code
            code_updates = encodings_one_hot.t() @ flat_inputs
            self.z_avg.data.mul_(self.ema_decay).add_(code_updates, alpha=1 - self.ema_decay)

            # Normalize and update the codebook
            n_eff = self.N + self.eps  # Avoid division by zero
            self.embeddings.data.copy_(self.z_avg / n_eff.unsqueeze(1))

            # Code re-initialization for underutilized codes
            if not self.no_random_restart:
                unused_codes = self.N < self.restart_thres
                if torch.any(unused_codes):
                    reinit_values = flat_inputs[torch.randperm(flat_inputs.shape[0])[: unused_codes.sum()]]
                    self.embeddings.data[unused_codes] = reinit_values

        embeddings_st = (embeddings - z).detach() + z

        return embeddings_st, encoding_indices, commitment_loss

    def dictionary_lookup(self, encodings):
        """Retrieve embeddings from codebook."""
        return F.embedding(encodings, self.embeddings)

    def soft_codebook_lookup(self, weights: torch.Tensor) -> torch.Tensor:
        """Perform soft lookup using weighted sum of embeddings."""
        return weights @ self.embeddings
