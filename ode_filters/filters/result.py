"""``FilterResult``: what every filter returns."""

from __future__ import annotations

from typing import NamedTuple

from jax import Array

from ..gaussian import Conditional, Gaussian


class StepRecord(NamedTuple):
    """The accepted steps of an adaptive solve, in order.

    Rows past ``count`` (or past the record's length, if the solve took more
    steps than it holds) are NaN.

    Attributes:
        t: Time at the end of each accepted step, shape ``[n]``.
        h: Size of each accepted step, shape ``[n]``.
        m: Filtered mean after each accepted step, shape ``[n, D]``.
        sigma_sqr: Diffusion estimate of each accepted step, shape ``[n, d]``.
        count: Number of accepted steps of the whole solve.
    """

    t: Array
    h: Array
    m: Array
    sigma_sqr: Array
    count: Array


class FilterResult(NamedTuple):
    """Result of a Gaussian ODE filter, in the original state coordinates.

    The time grid has ``K`` nodes; per-step arrays have ``K - 1`` entries, entry
    ``n`` belonging to the step from ``t[n]`` to ``t[n + 1]``. Covariances are
    square-root factors: ``P = P_sqr.T @ P_sqr``.

    Attributes:
        t: Time grid, shape ``[K]``.
        m: Filtered means, shape ``[K, D]``.
        P_sqr: Filtered square-root covariances, shape ``[K, D, D]``.
        log_likelihood: ODE-information evidence,
            ``sum_n log p(z_n | z_<n, y_<n)``, with the ODE residuals ``z``
            observed as zero, taken after the diffusion calibration (so not
            comparable across calibration policies). On the adaptive path it
            also contains the data evidence.
        log_likelihood_obs: Data evidence, ``sum_n log p(y_n | z_<=n, y_<n)``:
            each observation conditioned on the ODE information up to its own
            time only (``None`` without data, and on the adaptive path). The sum
            with ``log_likelihood`` is ``log p(Z, Y)``; it is not the
            conditional evidence ``log p(Y | Z)``, which also conditions on
            later ODE information (see ``fenrir_loglik``).
        sigma_sqr: Diffusion estimate per step, shape ``[K - 1]`` (``[K - 1, d]``
            for a per-component policy); ``None`` on the adaptive path.
        m_pred: Predicted means per step, ``[K - 1, D]`` (``None`` on the
            adaptive path).
        P_pred_sqr: Predicted square-root covariances per step.
        G_back: Gains of the backward conditionals ``p(x_n | x_{n+1})``, per
            step, ``[K - 1, D, D]``. On the adaptive path with
            ``smoother=True`` one composite conditional per save interval;
            ``None`` without.
        d_back: Offsets of the backward conditionals.
        P_back_sqr: Square-root covariances of the backward conditionals.
        mz: Means of the ODE-information innovations per step (the residual at
            the predicted mean).
        Pz_sqr: Square-root covariances of the ODE-information innovations.
        mz_obs: Means of the data innovations per step, ``h(m) - y`` at the
            state after the ODE update (``None`` without data). Rows without
            data at a step are zero.
        Pz_obs_sqr: Square-root covariances of the data innovations. Rows
            without data have unit variance and no correlation, so they add
            nothing to the whitened residual.
        success: Adaptive path: whether every save time was reached with a
            finite log-likelihood (``None`` on the fixed grid).
        n_accepted: Adaptive path: accepted steps per save interval.
        n_rejected: Adaptive path: rejected steps per save interval.
        steps: Adaptive path with ``record_steps > 0``: the accepted steps
            (see ``StepRecord``).
    """

    t: Array
    m: Array
    P_sqr: Array
    log_likelihood: Array
    log_likelihood_obs: Array | None = None
    sigma_sqr: Array | None = None
    m_pred: Array | None = None
    P_pred_sqr: Array | None = None
    G_back: Array | None = None
    d_back: Array | None = None
    P_back_sqr: Array | None = None
    mz: Array | None = None
    Pz_sqr: Array | None = None
    mz_obs: Array | None = None
    Pz_obs_sqr: Array | None = None
    success: Array | None = None
    n_accepted: Array | None = None
    n_rejected: Array | None = None
    steps: StepRecord | None = None

    @property
    def filtered(self) -> Gaussian:
        """The filtered marginals as a stacked ``Gaussian``."""
        return Gaussian(self.m, self.P_sqr)

    @property
    def backward(self) -> Conditional:
        """The backward conditionals as a stacked ``Conditional``.

        Raises:
            ValueError: If the result carries none (the adaptive path without
                ``smoother=True``).
        """
        if self.G_back is None or self.d_back is None or self.P_back_sqr is None:
            raise ValueError(
                "This result carries no backward conditionals: run "
                "gaussian_filter_adaptive with smoother=True, or use the fixed-grid "
                "gaussian_filter."
            )
        return Conditional(self.G_back, self.d_back, self.P_back_sqr)
