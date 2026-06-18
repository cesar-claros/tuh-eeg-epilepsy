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
    peak_freq_hz : float | None
        Frequency (Hz) of the strongest lobe of the dilated kernel's response,
        when a sampling rate was supplied during ranking; otherwise None.
    """

    rank: int
    count: float
    dilation: int
    representation: str
    group: int
    kernel: int
    weight: torch.Tensor
    peak_freq_hz: float | None = None


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
    peak_freq_hz : float | None
        Frequency (Hz) of the strongest lobe of the dilated kernel's response,
        when a sampling rate was supplied during ranking; otherwise None.
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
    peak_freq_hz: float | None = None


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
                 device=None,
                 ):
        super().__init__()

        self.device = self._resolve_device(device)
        self.n_timepoints = n_timepoints
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
        # Initialize weights on CPU (so the seeded RNG is device-independent),
        # then move to the target device.
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
        self.W = self.W.to(self.device)

        num_channels_per = np.clip(n_channels // 2, 2, max_num_channels)
        self.idx = torch.stack([
            torch.randint(
                0,
                n_channels,
                (self.divisor, self.h, num_channels_per),
                generator=generator,
            )
            for _ in range(self.num_dilations)
        ]).to(self.device)

        # Optional per-kernel count matrices of shape (num_dilations, divisor, h,
        # k). When track_counts is True, each forward pass accumulates, per
        # kernel: win FREQUENCY (win_counts_*, how often it wins) and win
        # MAGNITUDE (value_counts_*, summed winning response value) for the max
        # and min competitions. If labels are passed to forward, the counts are
        # also binned per class. These are pure diagnostics; they do not affect
        # the returned features.
        self.track_counts = track_counts
        shape = (self.num_dilations, self.divisor, self.h, self.k)
        self.win_counts_max = torch.zeros(shape, device=self.device)
        self.win_counts_min = torch.zeros(shape, device=self.device)
        self.value_counts_max = torch.zeros(shape, device=self.device)
        self.value_counts_min = torch.zeros(shape, device=self.device)
        self.win_counts_max_by_class: dict[int, torch.Tensor] = {}
        self.win_counts_min_by_class: dict[int, torch.Tensor] = {}
        self.value_counts_max_by_class: dict[int, torch.Tensor] = {}
        self.value_counts_min_by_class: dict[int, torch.Tensor] = {}

    @staticmethod
    def _resolve_device(device):
        """Resolve a device spec (None/"cpu"/"cuda"/"cuda:N"/"auto") to a torch.device."""
        if device is None or device == "cpu":
            return torch.device("cpu")
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def forward(self, X: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
        X = X.to(self.device, non_blocking=True)
        n_examples, n_channels, _ = X.shape

        # Precompute per-class example masks once (only when tracking with labels).
        class_masks = None
        if self.track_counts and y is not None:
            labels = y.reshape(-1).to(self.device, non_blocking=True)
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
                count_max = torch.zeros(n_examples, self.h, self.k, device=self.device)

                min_values, min_indices = _Z.min(2)
                count_min = torch.zeros(n_examples, self.h, self.k, device=self.device)

                count_max.scatter_add_(-1, max_indices, max_values)
                count_min.scatter_add_(-1, min_indices, torch.ones_like(min_values))

                if self.track_counts:
                    # count_max is the value-weighted (magnitude) max already;
                    # count_min is the hard (frequency) min already. Compute the
                    # two complements: frequency-max and magnitude-min.
                    win_max = torch.zeros(n_examples, self.h, self.k, device=self.device)
                    win_max.scatter_add_(-1, max_indices, torch.ones_like(max_values))
                    val_min = torch.zeros(n_examples, self.h, self.k, device=self.device)
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
            self.win_counts_max_by_class[c] = torch.zeros(shape, device=self.device)
            self.win_counts_min_by_class[c] = torch.zeros(shape, device=self.device)
            self.value_counts_max_by_class[c] = torch.zeros(shape, device=self.device)
            self.value_counts_min_by_class[c] = torch.zeros(shape, device=self.device)

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

    @staticmethod
    def kernel_frequency_response(
        weight: torch.Tensor, dilation: int, sfreq: float, n_freqs: int = 257,
        representation: str = "raw",
    ) -> tuple:
        """Magnitude frequency response of a dilated length-9 kernel on the signal x.

        The dilated kernel has taps at samples 0, d, ..., 8d; its own response at
        physical frequency f is ``H(f) = sum_j w_j exp(-i 2*pi (d f / sfreq) j)``.
        What this returns depends on which representation the kernel was applied to.

        ``representation='raw'``: the kernel convolves the signal x, so the response
        IS ``|H(f)|``. As a comb filter it repeats every ``sfreq / d`` Hz into ``d``
        identical replicas across [0, Nyquist], so only the principal band
        ``[0, sfreq / (2 d)]`` is returned (the higher replicas are exact copies) and
        ``peak_frequency`` reports the fundamental (the lowest tooth), avoiding an
        argmax over equal replicas. Zero-mean weights give no DC response.

        ``representation='diff'``: the kernel convolves the first difference
        ``dx_t = x_{t+1} - x_t``, so the EFFECTIVE filter on x is the kernel composed
        with the first-difference operator. The (undilated, lag-1) difference has
        response ``|1 - e^{-i w}| = 2|sin(pi f / sfreq)|`` (0 at DC, 2 at Nyquist), so
        the returned magnitude is ``|H(f)| * 2|sin(pi f / sfreq)|``. Because that
        high-pass is undilated it is NOT comb-periodic, so it breaks the equal-replica
        symmetry (it boosts higher replicas); the response is therefore returned over
        the FULL band ``[0, sfreq / 2]`` so the peak can land on the boosted high
        replica rather than the suppressed fundamental. This is the "View B" reading:
        every kernel's spectrum is its response to the raw signal, on one comparable
        Hz axis (see documentation/hydra_kernel_spectra.md).

        Parameters
        ----------
        weight : torch.Tensor
            The length-9 kernel weights.
        dilation : int
            The dilation the kernel is applied with.
        sfreq : float
            Sampling rate in Hz.
        n_freqs : int, default=257
            Number of frequency points across the returned band.
        representation : str, default='raw'
            'raw' (kernel on the signal) or 'diff' (kernel on the first difference);
            'diff' includes the first-difference high-pass and uses the full band.

        Returns
        -------
        tuple
            ``(freqs_hz, magnitude)`` as 1-D tensors of length ``n_freqs``. For 'raw'
            ``freqs_hz`` spans ``[0, sfreq / (2 * dilation)]``; for 'diff'
            ``[0, sfreq / 2]``.
        """
        fmax = sfreq / 2.0 if representation == "diff" else sfreq / (2.0 * dilation)
        freqs = torch.linspace(0.0, fmax, n_freqs, device=weight.device)
        taps = torch.arange(weight.numel(), dtype=weight.dtype, device=weight.device)
        phase = (2.0 * torch.pi * dilation / sfreq) * freqs[:, None] * taps[None, :]
        response = (weight[None, :] * torch.exp(-1j * phase)).sum(-1)
        mag = response.abs()
        if representation == "diff":
            # Effective response on x: multiply by the undilated first-difference
            # high-pass |1 - e^{-i w}| = 2|sin(pi f / sfreq)|.
            mag = mag * (2.0 * torch.sin(torch.pi * freqs / sfreq).abs())
        return freqs, mag

    @staticmethod
    def peak_frequency(
        weight: torch.Tensor, dilation: int, sfreq: float, n_freqs: int = 257,
        representation: str = "raw",
    ) -> float:
        """Peak frequency (Hz) of the kernel's response on the signal x.

        For 'raw' this is the fundamental (lowest comb tooth) in the principal band;
        for 'diff' it is the strongest lobe of the EFFECTIVE response on x (the kernel
        times the first-difference high-pass) over the full band, which tends to sit
        high because the high-pass boosts higher frequencies."""
        freqs, mag = HydraTransform.kernel_frequency_response(
            weight, dilation, sfreq, n_freqs, representation
        )
        return float(freqs[int(mag.argmax())].item())

    @staticmethod
    def spectral_centroid(
        weight: torch.Tensor, dilation: int, sfreq: float, n_freqs: int = 257,
        representation: str = "raw",
    ) -> float:
        """Magnitude-weighted mean frequency (Hz) of the kernel's response on x.

        The first-difference high-pass is included for 'diff' (the effective response
        on the signal); 'raw' uses the principal band (one comb period)."""
        freqs, mag = HydraTransform.kernel_frequency_response(
            weight, dilation, sfreq, n_freqs, representation
        )
        total = mag.sum()
        if float(total) == 0.0:
            return 0.0
        return float((freqs * mag).sum().item() / total.item())

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

    def _peak_freq(
        self, weight: torch.Tensor, dilation: int, sfreq: float | None,
        representation: str = "raw",
    ):
        """Peak frequency for a kernel, or None when no sampling rate is given."""
        if sfreq is None:
            return None
        return self.peak_frequency(weight, dilation, sfreq, representation=representation)

    def top_kernels(
        self,
        n_top: int,
        by: str = "max",
        class_label: int | None = None,
        weighting: str = "frequency",
        sfreq: float | None = None,
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
        sfreq : float | None, default=None
            If given, populate each kernel's ``peak_freq_hz`` (in Hz).

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
            dilation = int(self.dilations[dilation_index].item())
            weight = weights[dilation_index, diff_index, group, kernel].clone().cpu()
            infos.append(
                KernelInfo(
                    rank=rank,
                    count=value,
                    dilation=dilation,
                    representation="raw" if diff_index == 0 else "diff",
                    group=group,
                    kernel=kernel,
                    weight=weight,
                    peak_freq_hz=self._peak_freq(
                        weight, dilation, sfreq, "raw" if diff_index == 0 else "diff"
                    ),
                )
            )
        return infos

    def top_kernels_by_coef(
        self,
        coef,
        n_top: int,
        combine: str = "sum",
        sfreq: float | None = None,
        feature_std=None,
    ) -> list["KernelInfo"]:
        """Rank kernels by the trained linear classifier's weight on their features.

        Unlike ``top_kernels`` (data win counts) and the discriminative ranking
        (per-class win fractions), this ranks kernels by how much the fitted linear
        classifier relies on them, so it does NOT need ``track_counts``. The HYDRA
        feature vector has two columns per kernel, the max-win and min-win counts,
        laid out exactly as ``forward`` emits them
        (``(num_dilations, divisor, 2, h, k)`` flattened). ``coef`` is the
        classifier weight per feature column; each kernel's importance aggregates
        the absolute weight of its two columns.

        Parameters
        ----------
        coef : array-like
            Classifier weight per feature column, length
            ``num_dilations * divisor * 2 * h * k`` (e.g. ``LogisticRegression``'s
            ``coef_[0]``), in the same order as the features returned by ``forward``.
        n_top : int
            Number of top kernels to return.
        combine : str, default="sum"
            How to aggregate a kernel's two feature columns (max-win, min-win):
            "sum" of absolute weights or "max" of absolute weights.
        sfreq : float | None, default=None
            If given, populate each kernel's ``peak_freq_hz`` (in Hz).
        feature_std : array-like | None, default=None
            Optional per-column std of the features the classifier saw, to weight
            ``|coef|`` by the feature's spread. With the default sparse scaler the
            features are already standardized, so leaving this None (pure ``|coef|``)
            is the natural importance.

        Returns
        -------
        list[KernelInfo]
            The top kernels by classifier weight, highest importance first. The
            ``count`` field holds the aggregated absolute weight.
        """
        coef = torch.as_tensor(coef, dtype=torch.float32).reshape(-1)
        expected = self.num_dilations * self.divisor * 2 * self.h * self.k
        if coef.numel() != expected:
            raise ValueError(
                f"`coef` has {coef.numel()} entries, expected {expected} "
                f"(num_dilations*divisor*2*h*k)."
            )
        importance = coef.abs()
        if feature_std is not None:
            std = torch.as_tensor(feature_std, dtype=torch.float32).reshape(-1)
            if std.numel() != expected:
                raise ValueError("`feature_std` length must match `coef` length.")
            importance = importance * std
        # (num_dilations, divisor, 2, h, k): axis 2 holds the kernel's two feature
        # columns (max-win count, min-win count). Collapse it to a per-kernel score.
        importance = importance.reshape(
            self.num_dilations, self.divisor, 2, self.h, self.k
        )
        if combine == "sum":
            per_kernel = importance.sum(2)
        elif combine == "max":
            per_kernel = importance.amax(2)
        else:
            raise ValueError(f"`combine` must be 'sum' or 'max', got {combine!r}")

        # per_kernel is (num_dilations, divisor, h, k), the same layout as the
        # win-count matrices, so reuse the existing topk + decode path.
        weights = self.kernel_weights()
        _, divisor, h, k = per_kernel.shape
        flat = per_kernel.reshape(-1)
        top_values, top_indices = torch.topk(flat, min(n_top, flat.numel()))

        infos: list[KernelInfo] = []
        for rank, (value, index) in enumerate(
            zip(top_values.tolist(), top_indices.tolist())
        ):
            dilation_index, diff_index, group, kernel = self._decode_index(
                index, divisor, h, k
            )
            dilation = int(self.dilations[dilation_index].item())
            weight = weights[dilation_index, diff_index, group, kernel].clone().cpu()
            infos.append(
                KernelInfo(
                    rank=rank,
                    count=value,
                    dilation=dilation,
                    representation="raw" if diff_index == 0 else "diff",
                    group=group,
                    kernel=kernel,
                    weight=weight,
                    peak_freq_hz=self._peak_freq(
                        weight, dilation, sfreq, "raw" if diff_index == 0 else "diff"
                    ),
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
        sfreq: float | None = None,
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
        sfreq : float | None, default=None
            If given, populate each kernel's ``peak_freq_hz`` (in Hz).

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
        t = self._discriminative_tensors(by, classes, weighting, metric)
        magnitude = t["magnitude"]
        _, top_indices = torch.topk(magnitude, min(n_top, magnitude.numel()))
        return [
            self._make_disc_kernel(rank, int(index), t, sfreq)
            for rank, index in enumerate(top_indices.tolist())
        ]

    def top_discriminative_kernels_per_class(
        self,
        n_per_class: int,
        by: str = "max",
        classes: tuple | None = None,
        weighting: str = "frequency",
        metric: str = "difference",
        sfreq: float | None = None,
    ) -> dict:
        """Top ``n_per_class`` kernels favoring each of the two classes.

        Unlike ``top_discriminative_kernels`` (which ranks by absolute score and
        can be dominated by one class), this returns a balanced view: for each
        class, the kernels with the most extreme score toward it. The arguments
        mirror ``top_discriminative_kernels`` (``n_per_class`` replaces ``n_top``).

        Returns
        -------
        dict
            ``{class_a: [...], class_b: [...]}``, each list ordered
            most-favoring-first.
        """
        t = self._discriminative_tensors(by, classes, weighting, metric)
        score = t["score"]
        favors_a = t["favors_a"]
        a_idx = torch.nonzero(favors_a, as_tuple=True)[0]
        b_idx = torch.nonzero(~favors_a, as_tuple=True)[0]
        a_sorted = a_idx[torch.argsort(score[a_idx], descending=True)][:n_per_class]
        b_sorted = b_idx[torch.argsort(score[b_idx], descending=False)][:n_per_class]
        return {
            t["class_a"]: [
                self._make_disc_kernel(rank, int(index), t, sfreq)
                for rank, index in enumerate(a_sorted.tolist())
            ],
            t["class_b"]: [
                self._make_disc_kernel(rank, int(index), t, sfreq)
                for rank, index in enumerate(b_sorted.tolist())
            ],
        }

    def _discriminative_tensors(self, by, classes, weighting, metric):
        """Precompute flat per-kernel discriminative tensors for two classes."""
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
        _, divisor, h, k = score.shape
        return {
            "class_a": class_a,
            "class_b": class_b,
            "weights": self.kernel_weights(),
            "divisor": divisor,
            "h": h,
            "k": k,
            "score": score.reshape(-1),
            "magnitude": magnitude.reshape(-1),
            "frac_a": frac_a.reshape(-1),
            "frac_b": frac_b.reshape(-1),
            "favors_a": favors_a.reshape(-1),
        }

    def _make_disc_kernel(self, rank, index, t, sfreq):
        """Build one DiscriminativeKernel from a flat index and precomputed tensors."""
        dilation_index, diff_index, group, kernel = self._decode_index(
            index, t["divisor"], t["h"], t["k"]
        )
        dilation = int(self.dilations[dilation_index].item())
        weight = t["weights"][dilation_index, diff_index, group, kernel].clone().cpu()
        return DiscriminativeKernel(
            rank=rank,
            score=t["score"][index].item(),
            favors=t["class_a"] if bool(t["favors_a"][index]) else t["class_b"],
            fractions={
                t["class_a"]: t["frac_a"][index].item(),
                t["class_b"]: t["frac_b"][index].item(),
            },
            dilation=dilation,
            representation="raw" if diff_index == 0 else "diff",
            group=group,
            kernel=kernel,
            weight=weight,
            peak_freq_hz=self._peak_freq(
                weight, dilation, sfreq, "raw" if diff_index == 0 else "diff"
            ),
        )


if __name__ == "__main__":
    _ = HydraTransform()
