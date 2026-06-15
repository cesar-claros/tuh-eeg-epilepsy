"""IC-bag feature transformer: univariate HYDRA per IC, pooled across ICs.

Input is a padded bag of IC sources ``(B, K_max, T)`` (the data engine's
``signal_mode='ic_bag'`` output; zero rows are padding). Each IC is transformed
independently by a univariate HYDRA, then the per-IC feature vectors are pooled
over the (variable, masked) IC axis with permutation-invariant operators. This
sidesteps the fixed-aligned-channel requirement of multichannel HYDRA, so every IC
of every recording contributes without recombining the raw sources.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.components.hydra_transformer import HydraTransformer

_POOL_OPS = ("mean", "max", "std")


class ICBagTransformer(nn.Module):
    """Univariate HYDRA over each IC of a window, pooled across ICs.

    Parameters
    ----------
    n_kernels : int, default=8
        Kernels per group for the underlying (univariate) HYDRA.
    n_groups : int, default=64
        Groups per dilation for the underlying HYDRA.
    random_state : int | None, default=None
        Seed for the HYDRA random kernels (shared across all ICs; reproducible).
    device : str | None, default=None
        Device for the HYDRA convolutions ('cpu' | 'cuda' | 'cuda:N' | 'auto').
    pool : sequence of str, default=('mean', 'max', 'std')
        Pooling operators over the IC axis. 'max' is the focal-source detector
        ('does any IC look epileptiform?'). Output dim is ``len(pool) * d``.
    append_count : bool, default=False
        If True, append the per-window valid-IC count as one extra feature.
    n_jobs : int, default=1
        Threads for the underlying HYDRA.
    ic_chunk_size : int, default=32
        Number of IC-instances pushed through HYDRA at once. The per-IC pass
        reshapes to ``(B*K_max, 1, T)``; processing it in chunks of this size bounds
        the conv1d GPU memory (raise for speed if memory allows, lower on OOM).
    """

    def __init__(
        self,
        n_kernels: int = 8,
        n_groups: int = 64,
        random_state: int | None = None,
        device: str | None = None,
        pool: tuple[str, ...] = ("mean", "max", "std"),
        append_count: bool = False,
        n_jobs: int = 1,
        ic_chunk_size: int = 32,
    ) -> None:
        super().__init__()
        pool = tuple(pool)
        bad = [p for p in pool if p not in _POOL_OPS]
        if bad:
            raise ValueError(f"Unknown pool ops {bad}; choose from {_POOL_OPS}.")
        if not pool:
            raise ValueError("pool must contain at least one of mean/max/std.")
        if ic_chunk_size < 1:
            raise ValueError("ic_chunk_size must be >= 1.")
        self.pool = pool
        self.append_count = append_count
        self.ic_chunk_size = ic_chunk_size
        # Univariate HYDRA: each IC is a single-channel series (max_num_channels=1).
        self.hydra = HydraTransformer(
            n_kernels=n_kernels,
            n_groups=n_groups,
            max_num_channels=1,
            n_jobs=n_jobs,
            random_state=random_state,
            track_counts=False,
            device=device,
        )

    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:  # noqa: ARG002
        """Transform a batch of IC bags into pooled features.

        Parameters
        ----------
        X : torch.Tensor
            Shape ``(B, K_max, T)``; all-zero IC rows are treated as padding.
        y : torch.Tensor | None
            Unused (kept for the feature-extractor interface).

        Returns
        -------
        torch.Tensor
            Shape ``(B, len(pool) * d)`` (plus 1 if ``append_count``), where ``d``
            is the univariate-HYDRA feature dimension.
        """
        b, k_max, t = X.shape
        # Valid ICs are rows that are not all-zero (padding stays exactly zero).
        mask = X.abs().sum(dim=-1) > 0  # (B, K_max)
        # Reshaping to (B*K_max, 1, T) inflates the effective batch by K_max, so a
        # single HYDRA pass would blow up the conv1d intermediates on GPU. Chunk
        # the per-IC pass (each IC is independent, so this is exact) to bound peak
        # memory regardless of B and K_max.
        flat = X.reshape(b * k_max, 1, t)
        feat_chunks = [
            self.hydra(flat[s:s + self.ic_chunk_size])
            for s in range(0, flat.shape[0], self.ic_chunk_size)
        ]
        feats = torch.cat(feat_chunks, dim=0)  # (B*K_max, d)
        d = feats.shape[1]
        feats = feats.reshape(b, k_max, d)

        mask = mask.to(feats.device)
        valid = mask.sum(dim=1)  # (B,)
        denom = valid.clamp(min=1).unsqueeze(-1).to(feats.dtype)  # (B, 1)
        m = mask.unsqueeze(-1).to(feats.dtype)  # (B, K_max, 1)

        mean = (feats * m).sum(dim=1) / denom  # (B, d)
        pooled: list[torch.Tensor] = []
        for op in self.pool:
            if op == "mean":
                pooled.append(mean)
            elif op == "max":
                neg = torch.finfo(feats.dtype).min
                pooled.append(feats.masked_fill(~mask.unsqueeze(-1), neg).max(dim=1).values)
            elif op == "std":
                var = (((feats - mean.unsqueeze(1)) ** 2) * m).sum(dim=1) / denom
                pooled.append(var.clamp(min=0).sqrt())

        out = torch.cat(pooled, dim=1)  # (B, len(pool)*d)
        if self.append_count:
            out = torch.cat([out, valid.unsqueeze(1).to(out.dtype)], dim=1)
        # Empty bags (no valid ICs) -> zero feature vector (also fixes the -inf
        # that max-pool would otherwise produce).
        return out * (valid > 0).unsqueeze(1).to(out.dtype)
