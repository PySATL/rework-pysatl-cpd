# -*- coding: ascii -*-
"""Gaussian conjugate likelihood for Bayesian online change-point detection."""

__author__ = "Alexey Tatyanenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt
from scipy import stats

from pysatl_cpd.algorithms.online.bayesian.protocol.likelihood import ILikelihood


class GaussianConjugate(ILikelihood):
    """Normal-Inverse-Gamma conjugate likelihood with Student-t predictive density.

    Parameters
    ----------
    mu_0
        Prior mean.
    k_0
        Prior pseudo-count (must be > 0).
    alpha_0
        Prior shape (must be > 0).
    beta_0
        Prior scale (must be > 0).

    Raises
    ------
    ValueError
        If any of *k_0*, *alpha_0*, *beta_0* are non-positive.
    """

    def __init__(self, mu_0: float, k_0: float, alpha_0: float, beta_0: float) -> None:
        if k_0 <= 0:
            raise ValueError("k_0 must be positive")
        if alpha_0 <= 0:
            raise ValueError("alpha_0 must be positive")
        if beta_0 <= 0:
            raise ValueError("beta_0 must be positive")

        self._mu_0 = np.float64(mu_0)
        self._k_0 = np.float64(k_0)
        self._alpha_0 = np.float64(alpha_0)
        self._beta_0 = np.float64(beta_0)

        self._mu_params: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._k_params: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._alpha_params: npt.NDArray[np.float64] = np.array([], dtype=np.float64)
        self._beta_params: npt.NDArray[np.float64] = np.array([], dtype=np.float64)

    def update(self, observation: np.float64) -> None:
        """Update posterior sufficient statistics with a new observation.

        Mutates internal parameter arrays by appending new posterior values
        (prepended with the prior at index 0).

        Parameters
        ----------
        observation
            New observation to incorporate.
        """
        mu_to_update = self._mu_params
        k_to_update = self._k_params
        alpha_to_update = self._alpha_params
        beta_to_update = self._beta_params

        k_new = k_to_update + 1.0
        mu_new = (k_to_update * mu_to_update + observation) / k_new
        alpha_new = alpha_to_update + 0.5
        beta_update_term = (k_to_update * (observation - mu_to_update) ** 2) / (2.0 * k_new)
        beta_new = beta_to_update + beta_update_term

        self._mu_params = np.append(np.array([self._mu_0]), mu_new)
        self._k_params = np.append(np.array([self._k_0]), k_new)
        self._alpha_params = np.append(np.array([self._alpha_0]), alpha_new)
        self._beta_params = np.append(np.array([self._beta_0]), beta_new)

    def predict(self, observation: np.float64, window: int | None = None) -> npt.NDArray[np.float64]:
        """Return predictive log-probabilities under Student-t densities.

        First element is the prior predictive log-likelihood; remaining
        elements are posterior predictive log-likelihoods for the most
        recent *window* run-length states.

        Parameters
        ----------
        observation
            Observation to evaluate.
        window
            Maximum number of posterior states to consider.
            Defaults to all accumulated states.

        Returns
        -------
        npt.NDArray[np.float64]
            Array of log-probabilities.
        """
        df_prior = 2.0 * self._alpha_0
        scale_prior = np.sqrt(self._beta_0 * (self._k_0 + 1.0) / (self._alpha_0 * self._k_0 + 1e-12))
        prior_pred_loglik = stats.t.logpdf(x=observation, df=df_prior, loc=self._mu_0, scale=scale_prior)

        if window is None:
            window = len(self._alpha_params)

        post_pred_loglik = np.array([], dtype=np.float64)
        if self._mu_params.size > 0:
            df_post = 2.0 * self._alpha_params[:window]
            loc_post = self._mu_params[:window]
            scale_post = np.sqrt(
                self._beta_params[:window]
                * (self._k_params[:window] + 1.0)
                / (self._alpha_params[:window] * self._k_params[:window] + 1e-12)
            )
            post_pred_loglik = np.asarray(
                stats.t.logpdf(x=observation, df=df_post, loc=loc_post, scale=scale_post),
                dtype=np.float64,
            )

        return np.append(np.float64(prior_pred_loglik), post_pred_loglik)

    def clear(self) -> None:
        """Reset all posterior parameter arrays to empty.

        Returns
        -------
        None
        """
        self._mu_params = np.array([], dtype=np.float64)
        self._k_params = np.array([], dtype=np.float64)
        self._alpha_params = np.array([], dtype=np.float64)
        self._beta_params = np.array([], dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"GaussianConjugate(mu_0={float(self._mu_0)!r}, k_0={float(self._k_0)!r}, "
            f"alpha_0={float(self._alpha_0)!r}, beta_0={float(self._beta_0)!r})"
        )
