"""HYDRA random-kernel transform (torch internals)."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class KernelInfo:
    """A single HYDRA kernel and how strongly it won the per-group competition.

    Attributes
    ----------
    rank : int
        Position in the ranking (0 is the top kernel).
    count : float
        The ranking value: win frequency (weighting="frequency") or summed
        winning response magnitude (weighting="magnitude").
    dilation : int
        The dilation value used for this kernel.
    representation : str
        "raw" for the signal, "diff" for its first difference.
    group : int
        The group index within the representation.
    kernel : int
        The kernel index within the group.
    weight : torch.Tensor
        The length-9 kernel weights.
    """

    rank: int
    count: float
    dilation: int
    representation: str
    group: int
    kernel: int
    weight: torch.Tensor


@dataclass
class DiscriminativeKernel:
    """A kernel ranked by how differently it fires across two classes.

    Attributes
    ----------
    rank : int
        Position in the ranking (0 is the most discriminative).
    score : float
        The discriminative score under the chosen metric (see
        ``top_discriminative_kernels``). Its sign / magnitude meaning depends on
        the metric; ``favors`` summarises which class it points to.
    favors : object
        The class label this kernel wins for relatively more often.
    fractions : dict
        Per-class within-group win fraction, ``{class_label: fraction}``.
    dilation : int
        The dilation value used for this kernel.
    representation : str
        "raw" for the signal, "diff" for its first difference.
    group : int
        The group index within the representation.
    kernel : int
        The kernel index within the group.
    weight : torch.Tensor
        The length-9 kernel weights.
    """

    rank: int
    score: float
    favors: object
    fractions: dict
    dilation: int
    representation: str
    group: int
    kernel: int
    weight: torch.Tensor


class HydraTransform(nn.Module):
    """HYDRA torch internals."""

    def __init__(self,
                 n_timepoints:int = 75000,
                 n_channels:int = 17,
                 k:int = 8,
                 g:int = 64,
                 max_num_channels:int = 8,
                 seed:int = None,
                 track_counts: bool = False,
                 ):
        super().__init__()

        self.k = k  # num kernels per group
        self.g = g  # num groups

        max_exponent = np.log2((n_timepoints - 1) / (9 - 1))  # kernel length = 9

        self.dilations = 2 ** torch.arange(int(max_exponent) + 1)
        self.num_dilations = len(self.dilations)

        self.paddings = torch.div(
            (9 - 1) * self.dilations, 2, rounding_mode="floor"
        ).int()

        self.divisor = min(2, self.g)
        self.h = self.g // self.divisor
        # Use a dedicated RNG seeded with `seed` so kernels and channel selections
        # are reproducible for a given seed (and distinct across seeds), without
        # mutating the global torch RNG state. When `seed` is None the draws fall
        # back to the global RNG.
        generator = None
        if isinstance(seed, int):
            generator = torch.Generator()
            generator.manual_seed(seed)
        # Initialize weights
        self.W = torch.randn(
            self.num_dilations,
            self.divisor,
            self.k * self.h,
            1,
            9,
            generator=generator,
        )
        self.W = self.W - self.W.mean(-1, keepdims=True)
        self.W = self.W / self.W.abs().sum(-1, keepdims=True)

        num_channels_per = np.clip(n_channels // 2, 2, max_num_channels)
        self.idx = [
            torch.randint(
                0,
                n_channels,
                (self.divisor, self.h, num_channels_per),
                generator=generator,
            )
            for _ in range(self.num_dilations)
        ]

        # Optional per-kernel count matrices of shape (num_dilations, divisor, h,
        # k). When track_counts is True, each forward pass accumulates, per
        # kernel: win FREQUENCY (win_counts_*, how often it wins) and win
        # MAGNITUDE (value_counts_*, summed winning response value) for the max
        # and min competitions. If labels are passed to forward, the counts are
        # also binned per class. These are pure diagnostics; they do not affect
        # the returned features.
        self.track_counts = track_counts
        shape = (self.num_dilations, self.divisor, self.h, self.k)
        self.win_counts_max = torch.zeros(shape)
        self.win_counts_min = torch.zeros(shape)
        self.value_counts_max = torch.zeros(shape)
        self.value_counts_min = torch.zeros(shape)
        self.win_counts_max_by_class: dict[int, torch.Tensor] = {}
        self.win_counts_min_by_class: dict[int, torch.Tensor] = {}
        self.value_counts_max_by_class: dict[int, torch.Tensor] = {}
        self.value_counts_min_by_class: dict[int, torch.Tensor] = {}

    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        n_examples, n_channels, _ = X.shape

        # Precompute per-class example masks once (only when tracking with labels).
        class_masks = None
        if self.track_counts and y is not None:
            labels = y.reshape(-1)
            class_masks = {
                int(c): (labels == c) for c in torch.unique(labels).tolist()
            }
            for c in class_masks:
                self._ensure_class_bins(c)

        if self.divisor > 1:
            diff_X = torch.diff(X)

        Z = []

        for dilation_index in range(self.num_dilations):
            d = self.dilations[dilation_index].item()
            p = self.paddings[dilation_index].item()

            for diff_index in range(self.divisor):
                if n_channels > 1:  # Multivariate
                    _Z = F.conv1d(
                        (
                            X[:, self.idx[dilation_index][diff_index]].sum(2)
                            if diff_index == 0
                            else diff_X[
                                :, self.idx[dilation_index][diff_index]
                            ].sum(2)
                        ),
                        self.W[dilation_index][diff_index],
                        dilation=d,
                        padding=p,
                        groups=self.h,
                    ).view(n_examples, self.h, self.k, -1)
                else:  # Univariate
                    _Z = F.conv1d(
                        X if diff_index == 0 else diff_X,
                        self.W[dilation_index, diff_index],
                        dilation=d,
                        padding=p,
                    ).view(n_examples, self.h, self.k, -1)

                max_values, max_indices = _Z.max(2)
                count_max = torch.zeros(n_examples, self.h, self.k)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(n_examples, self.h, self.k)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                if self.track_counts:
                    # count_max is the value-weighted (magnitude) max already;
                    # count_min is the hard (frequency) min already. Compute the
                    # two complements: frequency-max and magnitude-min.
                    win_max = torch.zeros(n_examples, self.h, self.k)
                    win_max.scatter_add_(-1, max_indices, torch.ones_like(max_values))
                    val_min = torch.zeros(n_examples, self.h, self.k)
                    val_min.scatter_add_(-1, min_indices, min_values)
                    self._accumulate(
                        dilation_index, diff_index, None,
                        win_max.sum(0), count_min.sum(0),
                        count_max.sum(0), val_min.sum(0),
                    )
                    if class_masks is not None:
                        for c, mask in class_masks.items():
                            self._accumulate(
                                dilation_index, diff_index, c,
                                win_max[mask].sum(0), count_min[mask].sum(0),
                                count_max[mask].sum(0), val_min[mask].sum(0),
                            )

                Z.append(count_max)
                Z.append(count_min)

        Z = torch.cat(Z, 1).view(n_examples, -1)

        return Z

    def _accumulate(self, dil, diff, c, freq_max, freq_min, mag_max, mag_min):
        """Add one batch's per-kernel counts into the global (c=None) or class bins."""
        if c is None:
            self.win_counts_max[dil, diff] += freq_max
            self.win_counts_min[dil, diff] += freq_min
            self.value_counts_max[dil, diff] += mag_max
            self.value_counts_min[dil, diff] += mag_min
        else:
            self.win_counts_max_by_class[c][dil, diff] += freq_max
            self.win_counts_min_by_class[c][dil, diff] += freq_min
            self.value_counts_max_by_class[c][dil, diff] += mag_max
            self.value_counts_min_by_class[c][dil, diff] += mag_min

    def _ensure_class_bins(self, c: int) -> None:
        """Lazily create the per-class count matrices for class ``c``."""
        if c not in self.win_counts_max_by_class:
            shape = (self.num_dilations, self.divisor, self.h, self.k)
            self.win_counts_max_by_class[c] = torch.zeros(shape)
            self.win_counts_min_by_class[c] = torch.zeros(shape)
            self.value_counts_max_by_class[c] = torch.zeros(shape)
            self.value_counts_min_by_class[c] = torch.zeros(shape)

    def reset_counts(self) -> None:
        """Zero the global and per-class count matrices."""
        for store in (self.win_counts_max, self.win_counts_min,
                      self.value_counts_max, self.value_counts_min):
            store.zero_()
        for store in (self.win_counts_max_by_class, self.win_counts_min_by_class,
                      self.value_counts_max_by_class, self.value_counts_min_by_class):
            store.clear()

    def class_labels(self) -> list:
        """Return the class labels seen during tracking, sorted."""
        return sorted(self.win_counts_max_by_class.keys())

    def kernel_weights(self) -> torch.Tensor:
        """Return all kernels reshaped to (num_dilations, divisor, h, k, 9)."""
        return self.W.reshape(self.num_dilations, self.divisor, self.h, self.k, 9)

    def _count_matrix(
        self, by: str, class_label: int | None = None, weighting: str = "frequency"
    ) -> torch.Tensor:
        """Select the (num_dilations, divisor, h, k) count matrix to rank by.

        ``weighting`` picks frequency (how often a kernel wins) vs magnitude (its
        summed winning response value); ``by`` picks the max / min / total
        competition; ``class_label`` picks a per-class matrix or the global one.
        """
        if weighting == "frequency":
            glob_max, glob_min = self.win_counts_max, self.win_counts_min
            cls_max, cls_min = self.win_counts_max_by_class, self.win_counts_min_by_class
        elif weighting == "magnitude":
            glob_max, glob_min = self.value_counts_max, self.value_counts_min
            cls_max = self.value_counts_max_by_class
            cls_min = self.value_counts_min_by_class
        else:
            raise ValueError(
                f"`weighting` must be 'frequency' or 'magnitude', got {weighting!r}"
            )

        if class_label is None:
            store_max, store_min = glob_max, glob_min
        elif class_label in cls_max:
            store_max, store_min = cls_max[class_label], cls_min[class_label]
        else:
            raise KeyError(
                f"No counts for class {class_label!r}; seen {self.class_labels()}"
            )

        if by == "max":
            return store_max
        if by == "min":
            return store_min
        if by == "total":
            return store_max + store_min
        raise ValueError(f"`by` must be 'max', 'min', or 'total', got {by!r}")

    @staticmethod
    def _decode_index(index: int, divisor: int, h: int, k: int) -> tuple:
        """Map a flat count-matrix index to (dilation, diff, group, kernel)."""
        block = divisor * h * k
        dilation_index = index // block
        within = index % block
        diff_index = within // (h * k)
        within_repr = within % (h * k)
        return dilation_index, diff_index, within_repr // k, within_repr % k

    @staticmethod
    def _within_group_fraction(counts: torch.Tensor) -> torch.Tensor:
        """Normalize counts so each group's kernels share sums to 1 (win fractions).

        Dividing by the signed group total keeps fractions in [0, 1] for both
        non-negative frequency counts and (possibly negative) magnitude counts.
        """
        totals = counts.sum(-1, keepdim=True)
        safe = torch.where(totals.abs() < 1e-12, torch.ones_like(totals), totals)
        return counts / safe

    @staticmethod
    def _discriminative_score(frac_a, frac_b, metric: str, eps: float = 1e-6):
        """Return (score, magnitude, favors_a) for two classes' win fractions.

        ``magnitude`` is what kernels are ranked by; ``favors_a`` is a boolean
        tensor that is True where the kernel favors the first class.
        """
        if metric == "difference":
            score = frac_a - frac_b
            return score, score.abs(), score >= 0
        if metric == "ratio":
            ratio = (frac_a + eps) / (frac_b + eps)
            return ratio, ratio.log().abs(), ratio >= 1.0
        if metric == "logodds":
            logit_a = ((frac_a + eps) / (1.0 - frac_a + eps)).log()
            logit_b = ((frac_b + eps) / (1.0 - frac_b + eps)).log()
            score = logit_a - logit_b
            return score, score.abs(), score >= 0
        raise ValueError(
            f"`metric` must be 'difference', 'ratio', or 'logodds', got {metric!r}"
        )

    def top_kernels(
        self,
        n_top: int,
        by: str = "max",
        class_label: int | None = None,
        weighting: str = "frequency",
    ) -> list[KernelInfo]:
        """Return the top kernels by win count and their weights.

        Requires count tracking (``track_counts=True``) and at least one forward
        pass.

        Parameters
        ----------
        n_top : int
            Number of top kernels to return.
        by : str, default="max"
            Which competition to rank by: "max", "min", or "total".
        class_label : int | None, default=None
            If given, rank by that class's counts instead of the global ones.
        weighting : str, default="frequency"
            "frequency" (how often a kernel wins) or "magnitude" (summed winning
            response value).

        Returns
        -------
        list[KernelInfo]
            The top kernels, highest count first.
        """
        counts = self._count_matrix(by, class_label, weighting)
        weights = self.kernel_weights()
        _, divisor, h, k = counts.shape
        flat = counts.reshape(-1)
        top_values, top_indices = torch.topk(flat, min(n_top, flat.numel()))

        infos: list[KernelInfo] = []
        for rank, (value, index) in enumerate(
            zip(top_values.tolist(), top_indices.tolist())
        ):
            dilation_index, diff_index, group, kernel = self._decode_index(
                index, divisor, h, k
            )
            infos.append(
                KernelInfo(
                    rank=rank,
                    count=value,
                    dilation=int(self.dilations[dilation_index].item()),
                    representation="raw" if diff_index == 0 else "diff",
                    group=group,
                    kernel=kernel,
                    weight=weights[dilation_index, diff_index, group, kernel].clone(),
                )
            )
        return infos

    def top_discriminative_kernels(
        self,
        n_top: int,
        by: str = "max",
        classes: tuple | None = None,
        weighting: str = "frequency",
        metric: str = "difference",
    ) -> list[DiscriminativeKernel]:
        """Return the kernels whose win rate differs most between two classes.

        Each class's counts are normalized to within-group win fractions (so the
        comparison is fair regardless of how many windows each class has), then
        combined by ``metric`` and ranked by the magnitude of the result.

        Requires per-class tracking: forward must have been called with labels
        and ``track_counts=True``.

        Parameters
        ----------
        n_top : int
            Number of top kernels to return.
        by : str, default="max"
            Which competition to rank by: "max", "min", or "total".
        classes : tuple | None, default=None
            The two class labels to contrast, as ``(a, b)``. Defaults to the two
            observed classes (sorted); required if more than two classes exist.
        weighting : str, default="frequency"
            "frequency" or "magnitude" counts (see ``top_kernels``).
        metric : str, default="difference"
            How to combine the two classes' win fractions: "difference"
            (``frac_a - frac_b``), "ratio" (``frac_a / frac_b``), or "logodds"
            (``logit(frac_a) - logit(frac_b)``).

        Returns
        -------
        list[DiscriminativeKernel]
            The most discriminative kernels first.

        Raises
        ------
        RuntimeError
            If no per-class counts have been accumulated.
        ValueError
            If ``classes`` is not given and there are not exactly two classes.
        """
        labels = self.class_labels()
        if not labels:
            raise RuntimeError(
                "No per-class counts. Run forward with labels and track_counts=True."
            )
        if classes is None:
            if len(labels) != 2:
                raise ValueError(
                    f"Pass `classes` to choose two of the seen labels {labels}."
                )
            classes = (labels[0], labels[1])
        class_a, class_b = classes

        frac_a = self._within_group_fraction(self._count_matrix(by, class_a, weighting))
        frac_b = self._within_group_fraction(self._count_matrix(by, class_b, weighting))
        score, magnitude, favors_a = self._discriminative_score(frac_a, frac_b, metric)

        weights = self.kernel_weights()
        _, divisor, h, k = score.shape
        flat_score = score.reshape(-1)
        flat_mag = magnitude.reshape(-1)
        flat_favors = favors_a.reshape(-1)
        flat_a = frac_a.reshape(-1)
        flat_b = frac_b.reshape(-1)
        _, top_indices = torch.topk(flat_mag, min(n_top, flat_mag.numel()))

        results: list[DiscriminativeKernel] = []
        for rank, index in enumerate(top_indices.tolist()):
            dilation_index, diff_index, group, kernel = self._decode_index(
                index, divisor, h, k
            )
            results.append(
                DiscriminativeKernel(
                    rank=rank,
                    score=flat_score[index].item(),
                    favors=class_a if bool(flat_favors[index]) else class_b,
                    fractions={class_a: flat_a[index].item(), class_b: flat_b[index].item()},
                    dilation=int(self.dilations[dilation_index].item()),
                    representation="raw" if diff_index == 0 else "diff",
                    group=group,
                    kernel=kernel,
                    weight=weights[dilation_index, diff_index, group, kernel].clone(),
                )
            )
        return results


if __name__ == "__main__":
    _ = HydraTransform()
