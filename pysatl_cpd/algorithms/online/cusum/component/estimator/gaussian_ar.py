# -*- coding: ascii -*-
"""Gaussian autoregressive estimating schema."""

from collections.abc import Sequence

import numpy as np

from pysatl_cpd.algorithms.online.cusum.abstracts.estimator import IEstimatingSchema, ISchemaEstimates
from pysatl_cpd.algorithms.online.cusum.utils import coerce_observation
from pysatl_cpd.typedefs import UnivariateNumericArray

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

try:
    from arch.univariate import ARX

    HAS_ARCH = True
except ImportError:  # pragma: no cover - environment dependent
    HAS_ARCH = False


class EstimatesGaussianAR(ISchemaEstimates):
    intercept: float
    coefficients: UnivariateNumericArray
    noise_variance: float
    history: UnivariateNumericArray


class GaussianARSchema(IEstimatingSchema[UnivariateNumericArray, EstimatesGaussianAR]):
    """Univariate autoregressive estimation schema backed by ``arch`` ARX.

    Parameters
    ----------
    autoreg_order
        Number of lags in the AR model (> 0).
    adaptive
        Whether to re-fit the model online after training.
    window
        Maximum number of observations retained for fitting.
        ``None`` means unbounded.

    Raises
    ------
    ValueError
        If *autoreg_order* is non-positive, or *window* is non-positive.
    RuntimeError
        If the optional ``arch`` dependency is not installed.
    """

    def __init__(self, autoreg_order: int = 1, adaptive: bool = True, window: int | None = None) -> None:
        if autoreg_order <= 0:
            raise ValueError(f"autoreg_order must be positive, got {autoreg_order}")
        if window is not None and window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if not HAS_ARCH:
            raise RuntimeError("Autoregressive CUSUM requires the optional 'arch' dependency")

        self.autoreg_order = autoreg_order
        self.adaptive = adaptive
        self.window = window

        self._data = np.array([], dtype=np.float64)
        self._intercept = 0.0
        self._coefficients = np.array([], dtype=np.float64)
        self._noise_variance = 0.0
        self._history = np.array([], dtype=np.float64)

    def _trim_data(self) -> None:
        if self.window is not None and len(self._data) > self.window:
            self._data = self._data[-self.window :]

    def _coerce_scalar(self, observation: UnivariateNumericArray) -> float:
        obs = coerce_observation(observation)
        if obs.shape[0] != 1:
            raise ValueError(f"GaussianARSchema only supports dim=1, got shape {obs.shape}")
        return float(obs.item())

    def _fit_weights(self) -> None:
        if len(self._data) <= self.autoreg_order:
            raise ValueError(
                "Not enough observations to fit autoregressive model: "
                f"need more than autoreg_order={self.autoreg_order}, got {len(self._data)}"
            )

        model = ARX(self._data, lags=self.autoreg_order, rescale=False)
        result = model.fit(disp="off")
        params = result.params

        self._intercept = float(params.get("Const", 0.0))
        self._coefficients = np.array(
            [float(params[f"y[{lag}]"]) for lag in range(1, self.autoreg_order + 1)],
            dtype=np.float64,
        )
        self._noise_variance = float(params["sigma2"])
        self._history = self._data[-self.autoreg_order :].copy()

    def train(self, train_set: Sequence[UnivariateNumericArray]) -> None:
        """Fit AR model parameters from a training sample.

        Parameters
        ----------
        train_set
            Training observations.
        """
        self._data = np.array([self._coerce_scalar(observation) for observation in train_set], dtype=np.float64)
        self._trim_data()
        self._fit_weights()

    def update(self, observation: UnivariateNumericArray) -> None:
        """Update the internal data buffer and optionally re-fit AR weights.

        Parameters
        ----------
        observation
            New observation.
        """
        value = self._coerce_scalar(observation)
        self._data = np.append(self._data, value)
        self._trim_data()

        if self.adaptive:
            self._fit_weights()

    @property
    def estimates(self) -> EstimatesGaussianAR:
        """Current AR model estimates.

        Returns
        -------
        EstimatesGaussianAR
        """
        return {
            "intercept": self._intercept,
            "coefficients": self._coefficients.copy(),
            "noise_variance": self._noise_variance,
            "history": self._history.copy(),
        }

    def reset(self) -> None:
        """Reset all AR parameters and data buffer.

        Returns
        -------
        None
        """
        self._data = np.array([], dtype=np.float64)
        self._intercept = 0.0
        self._coefficients = np.array([], dtype=np.float64)
        self._noise_variance = 0.0
        self._history = np.array([], dtype=np.float64)
