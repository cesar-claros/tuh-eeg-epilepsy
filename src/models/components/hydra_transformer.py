"""Hydra Transformer."""

__maintainer__ = ["TonyBagnall"]
__all__ = ["HydraTransformer"]

from torch import nn
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from aeon.transformations.collection import BaseCollectionTransformer
from aeon.utils.validation import check_n_jobs
from aeon.utils.validation._dependencies import _check_soft_dependencies
from src.models.components.hydra_transform import (
    DiscriminativeKernel,
    HydraTransform,
    KernelInfo,
)

class HydraTransformer(nn.Module):
    """Hydra Transformer.

    The algorithm utilises convolutional kernels grouped into ``g`` groups per dilation
    with ``k`` kernels per group. It transforms input time series using these kernels
    and counts the kernels representing the closest match to the input at each time
    point. This counts for each group are then concatenated and returned.

    The algorithm combines aspects of both Rocket (convolutional approach)
    and traditional dictionary methods (pattern counting), It extracts features from
    both the base series and first-order differences of the series.

    Parameters
    ----------
    n_kernels : int, default=8
        Number of kernels per group.
    n_groups : int, default=64
        Number of groups per dilation.
    max_num_channels : int, default=8
        Maximum number of channels to use for each dilation.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, default=None
        If `int`, random_state is the seed used by the random number generator;
        If `RandomState` instance, random_state is the random number generator;
        If `None`, the random number generator is the `RandomState` instance used
        by `np.random`.
    output_type : str, default='tensor'
        The output type of the transformer.
        Can be either 'tensor' or 'numpy' or 'dataframe'.
        If 'tensor', the output will be a PyTorch tensor. If 'numpy', the output
        will be a NumPy array. If 'dataframe', the output will be a pandas DataFrame.


    See Also
    --------
    HydraClassifier
    MultiRocketHydraClassifier

    Notes
    -----
    Original code: https://github.com/angus924/hydra

    References
    ----------
    .. [1] Dempster, A., Schmidt, D.F. and Webb, G.I., 2023. Hydra: Competing
        convolutional kernels for fast and accurate time series classification.
        Data Mining and Knowledge Discovery, pp.1-27.

    Examples
    --------
    >>> from aeon.transformations.collection.convolution_based import HydraTransformer
    >>> from aeon.testing.data_generation import make_example_3d_numpy
    >>> X, _ = make_example_3d_numpy(n_cases=10, n_channels=1, n_timepoints=12,
    ...                              random_state=0)
    >>> clf = HydraTransformer(random_state=0)  # doctest: +SKIP
    >>> clf.fit(X)  # doctest: +SKIP
    HydraTransformer(random_state=0)
    >>> clf.transform(X)[0]  # doctest: +SKIP
    tensor([0.6077, 1.3868, 0.2571,  ..., 1.0000, 1.0000, 2.0000])
    """

    _tags = {
        "capability:multivariate": True,
        "capability:multithreading": True,
        "output_data_type": "Tabular",
        "algorithm_type": "convolution",
        "python_dependencies": "torch",
    }

    def __init__(
        self,
        n_kernels=8,
        n_groups=64,
        max_num_channels=8,
        n_jobs=1,
        random_state=None,
        track_counts=False,
    ):
        super().__init__()
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.max_num_channels = max_num_channels
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.track_counts = track_counts
        self._n_jobs = check_n_jobs(self.n_jobs)
        self._hydra = None


    def _lazy_setup(self, X:torch.Tensor) -> None:
        """Set up HydraTransform instance lazily based on input data shape."""
        torch.set_num_threads(self._n_jobs)
        self._hydra = HydraTransform(
            n_timepoints = X.shape[2],
            n_channels = X.shape[1],
            k = self.n_kernels,
            g = self.n_groups,
            max_num_channels = self.max_num_channels,
            seed = self.random_state,
            track_counts = self.track_counts,
        )

    def forward(self, X:torch.Tensor, y:torch.Tensor | None = None) -> torch.Tensor:
        if self._hydra is None:
            self._lazy_setup(X)
        # `y` (optional class labels) lets the transform bin win counts per class
        # when track_counts is True; it never affects the returned features.
        transformed = self._hydra(X, y)
        return transformed

    def reset_counts(self) -> None:
        """Reset the underlying global and per-class kernel win-count matrices."""
        if self._hydra is not None:
            self._hydra.reset_counts()

    def class_labels(self) -> list:
        """Return the class labels seen during tracking, sorted."""
        if self._hydra is None:
            return []
        return self._hydra.class_labels()

    def top_kernels(
        self,
        n_top: int,
        by: str = "max",
        class_label: int | None = None,
        weighting: str = "frequency",
    ) -> list[KernelInfo]:
        """Return the top winning kernels from the underlying HydraTransform.

        Requires this transformer to have been built with ``track_counts=True``
        and to have processed at least one batch. Pass ``class_label`` to rank by
        a single class's counts, and ``weighting`` to choose frequency vs
        magnitude counts.
        """
        if self._hydra is None:
            raise RuntimeError("Call the transformer on data before top_kernels().")
        return self._hydra.top_kernels(
            n_top, by=by, class_label=class_label, weighting=weighting
        )

    def top_discriminative_kernels(
        self,
        n_top: int,
        by: str = "max",
        classes: tuple | None = None,
        weighting: str = "frequency",
        metric: str = "difference",
    ) -> list[DiscriminativeKernel]:
        """Return the kernels whose win rate differs most between two classes.

        Requires per-class tracking: the transformer must have been built with
        ``track_counts=True`` and processed labelled batches (``forward(X, y)``).
        ``weighting`` and ``metric`` select the count weighting and the
        class-comparison metric (see ``HydraTransform.top_discriminative_kernels``).
        """
        if self._hydra is None:
            raise RuntimeError(
                "Call the transformer on labelled data before top_discriminative_kernels()."
            )
        return self._hydra.top_discriminative_kernels(
            n_top, by=by, classes=classes, weighting=weighting, metric=metric
        )

class _SparseScaler:
    """Sparse Scaler for hydra transform."""

    def __init__(self, mask=True, exponent=4):
        self.mask = mask
        self.exponent = exponent

    def fit(self, X, y=None):
        X = X.clamp(0).sqrt()

        self.epsilon = (X == 0).float().mean(0) ** self.exponent + 1e-8

        self.mu = X.mean(0)
        self.sigma = X.std(0) + self.epsilon

    def transform(self, X, y=None):
        X = X.clamp(0).sqrt()

        if self.mask:
            return ((X - self.mu) * (X != 0)) / self.sigma
        else:
            return (X - self.mu) / self.sigma

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)