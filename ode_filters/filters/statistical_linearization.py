"""Statistical linear regression (SLR) of an ODE-information residual.

Every Gaussian filter replaces the nonlinear residual ``g`` by the affine map
that best fits it *under some Gaussian*. Which Gaussian, and how the required
expectations are computed, is the entire difference between EK1, the UKF, the
statistically linearised filter (SLF), and their iterated variants. This module
implements the expectation side of that choice by quadrature.

**Only the vector field's arguments are integrated over.** The residual
``g(X) = E_c X - f(E_args X)`` is affine in ``X`` except through ``E_args X``, so
by Stein's identity the SLR reduces to expectations under the *projected*
marginal ``N(E_args m, E_args P E_args^T)``::

    H     = E[jacobian_g]                     (= E_c - E[J_f] E_args)
    c     = E[g] - H m
    Omega = Var[g] - H P H^T = Var[g(X) - H X - c]

The quadrature therefore runs in ``p = E_args.shape[0]`` dimensions -- the ODE
dimension -- and never in the ``d (q+1)`` state dimension. Two consequences:

- the tensor Gauss-Hermite cost ``m^p`` is indexed by the ODE dimension, not by
  the state dimension, so the spectrally accurate rule is affordable here;
- no matrix square root of the full ``P`` is ever taken. Sigma-point filters are
  reported unstable for ``q >= 3`` at small step sizes (Tronarp, Kersting,
  Sarkka & Hennig 2019, sec. 5) because ``cond(P) ~ h^(-2q)`` in the derivative
  coordinates. Factoring only the ``p x p`` projected block avoids that entirely.

``Omega`` is returned as a square-root factor built from the weighted fit
residuals at the nodes, so it is PSD by construction and needs no Cholesky of a
possibly-singular matrix. This requires **non-negative quadrature weights**,
which is why the rules here are Gauss-Hermite and spherical cubature rather than
the classic scaled unscented transform (whose ``beta`` term puts a negative
weight on the centre point). ``Omega`` vanishes identically for an affine vector
field, so an SLR correction is a no-op on linear problems.

References:
    Sarkka (2013), *Bayesian Filtering and Smoothing*, ch. 5-6 (SLF / sigma-point
    filters as one moment-matching family).
    Tronarp, Kersting, Sarkka & Hennig (2019), *Probabilistic solutions to ODEs
    as nonlinear Bayesian filtering*, sec. 2.3-2.5 (the exact filter moments for
    the ODE residual, and the taxonomy of ways to approximate them).
    Kersting & Hennig (2016), *Active Uncertainty Calibration in Bayesian ODE
    Solvers* (Bayesian-quadrature computation of the same two moments).
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as np
import numpy as onp
from jax import Array
from jax.typing import ArrayLike

from ..inference.sqr_gaussian_inference import sqr_marginalization
from ..measurement.measurement_models import BaseODEInformation

#: Quadrature rules selectable by name.
QuadratureRule = str

#: Guard on the tensor-product node count. Tensor Gauss-Hermite costs ``m^p``
#: nodes; past this the caller should switch to ``"cubature"`` (``2 p`` nodes).
MAX_NODES = 4096


class SLRModel(NamedTuple):
    """Affine surrogate of the residual produced by statistical linearisation.

    Attributes:
        H: Effective observation matrix, shape ``[obs_dim, state_dim]``. Equals
            the expected Jacobian of ``g`` under the linearisation density; the
            EK1 Jacobian is its zero-spread limit.
        c: Effective offset, shape ``[obs_dim]``, so that ``H x + c`` has the
            correct mean under the linearisation density.
        R_eff_sqr: Square root of the *effective* measurement noise
            ``R + Omega``, shape ``[obs_dim, obs_dim]``, upper triangular
            (``R_eff = R_eff_sqr.T @ R_eff_sqr``). This is what the update
            consumes: the linearisation residual enters the innovation
            covariance as inflated measurement noise, which is exactly how a
            sigma-point filter's ``S`` differs from the EKF's ``H P H^T + R``.
        Omega_sqr: Square-root factor of the linearisation-residual covariance
            alone, shape ``[n_nodes, obs_dim]`` (``Omega = Omega_sqr.T @
            Omega_sqr``). Not triangular and not square -- it is the stacked
            weighted fit residuals at the quadrature nodes. Exposed for
            diagnostics; ``R_eff_sqr`` already contains it. Zero for an affine
            vector field.
    """

    H: Array
    c: Array
    R_eff_sqr: Array
    Omega_sqr: Array


def gauss_hermite_rule(dim: int, n_nodes: int) -> tuple[Array, Array]:
    """Tensor-product Gauss-Hermite nodes and weights for a standard normal.

    Uses the probabilists' Hermite rule, so the nodes are in units of standard
    deviations and the weights sum to one. A rule with ``n_nodes`` points per
    dimension integrates polynomials up to total degree ``2 * n_nodes - 1``
    exactly.

    Args:
        dim: Dimension of the integration domain (``p``).
        n_nodes: Nodes per dimension (``m``); the rule has ``m ** dim`` nodes.

    Returns:
        Tuple ``(nodes, weights)`` of shapes ``[m ** dim, dim]`` and
        ``[m ** dim]``. ``weights`` are strictly positive and sum to one.

    Raises:
        ValueError: If ``n_nodes < 1`` or the node count exceeds
            :data:`MAX_NODES`.
    """
    if n_nodes < 1:
        raise ValueError(f"n_nodes must be >= 1, got {n_nodes!r}.")
    total = n_nodes**dim
    if total > MAX_NODES:
        raise ValueError(
            f"Tensor Gauss-Hermite with n_nodes={n_nodes} in dim={dim} needs "
            f"{total} nodes, above MAX_NODES={MAX_NODES}. Use "
            'rule="cubature" (2 * dim nodes) or lower n_nodes.'
        )
    x_1d, w_1d = onp.polynomial.hermite_e.hermegauss(n_nodes)
    w_1d = w_1d / onp.sqrt(2.0 * onp.pi)  # probabilists' normalisation: sum = 1

    grids = onp.meshgrid(*([x_1d] * dim), indexing="ij")
    nodes = onp.stack([g.reshape(-1) for g in grids], axis=-1)
    w_grids = onp.meshgrid(*([w_1d] * dim), indexing="ij")
    weights = onp.prod(onp.stack([g.reshape(-1) for g in w_grids], axis=-1), axis=-1)
    return np.asarray(nodes), np.asarray(weights)


def cubature_rule(dim: int) -> tuple[Array, Array]:
    """Third-degree spherical-radial cubature nodes and weights.

    The ``2 * dim`` points ``+- sqrt(dim) e_j`` with equal weights ``1 / (2 dim)``
    (Arasaratnam & Haykin 2009). Exact for polynomials up to total degree three,
    dimension-linear in cost, and all weights positive -- unlike the scaled
    unscented transform, which is the same third-degree rule plus a centre point
    whose covariance weight can be negative.

    Args:
        dim: Dimension of the integration domain (``p``).

    Returns:
        Tuple ``(nodes, weights)`` of shapes ``[2 * dim, dim]`` and ``[2 * dim]``.
    """
    eye = onp.eye(dim)
    nodes = onp.sqrt(dim) * onp.concatenate([eye, -eye], axis=0)
    weights = onp.full(2 * dim, 1.0 / (2 * dim))
    return np.asarray(nodes), np.asarray(weights)


def quadrature_nodes(
    rule: QuadratureRule, dim: int, n_nodes: int
) -> tuple[Array, Array]:
    """Dispatch to a named quadrature rule.

    Args:
        rule: ``"gauss_hermite"`` (tensor product, ``n_nodes`` per dimension) or
            ``"cubature"`` (third-degree spherical, ``n_nodes`` ignored).
        dim: Dimension of the integration domain.
        n_nodes: Nodes per dimension for the Gauss-Hermite rule.

    Returns:
        Tuple ``(nodes, weights)`` in standard-normal coordinates.

    Raises:
        ValueError: If ``rule`` is not a known rule name.
    """
    if rule == "gauss_hermite":
        return gauss_hermite_rule(dim, n_nodes)
    if rule == "cubature":
        return cubature_rule(dim)
    raise ValueError(
        f'rule must be "gauss_hermite" or "cubature", got {rule!r}. The scaled '
        "unscented transform is deliberately not offered: its negative centre "
        "weight breaks the square-root construction of Omega."
    )


def _arg_projection(measure: BaseODEInformation, state_dim: int) -> Array:
    """Selection matrix for the coordinates the residual is nonlinear in.

    Falls back to the identity for measurement models that do not declare
    ``E_args`` (e.g. :class:`~ode_filters.measurement.measurement_models.\
BlackBoxMeasurement`, which is duck-typed rather than a
    :class:`BaseODEInformation` subclass). That fallback is always *correct* --
    quadrature over the full state integrates over a superset of the directions
    that matter -- but it costs the full state dimension.
    """
    proj = getattr(measure, "E_args", None)
    if proj is None:
        return np.eye(state_dim)
    return np.asarray(proj)


def slr_linearize(
    measure: BaseODEInformation,
    m_lin: Array,
    P_lin_sqr: Array,
    *,
    t: ArrayLike = 0.0,
    rule: QuadratureRule = "gauss_hermite",
    n_nodes: int = 4,
) -> SLRModel:
    """Statistically linearise the residual with respect to ``N(m_lin, P_lin)``.

    Computes the MSE-optimal affine surrogate of ``measure.g`` under the given
    Gaussian, with the required expectations evaluated by quadrature over the
    *projected* marginal ``N(E_args m_lin, E_args P_lin E_args^T)``. See the
    module docstring for why that projection is exact.

    The linearisation density is a free argument, and it is the axis that
    separates the filter variants: passing the *predictive* moments gives the
    one-shot sigma-point / SLF update, passing the current *posterior* moments
    gives the posterior-linearisation iteration (IPLF), and passing *smoothed*
    moments gives IPLS.

    Args:
        measure: ODE-information measurement model. Its ``E_args`` declares which
            coordinates the vector field reads; use
            :func:`check_arg_projection` when writing a custom subclass.
        m_lin: Mean of the linearisation density, shape ``[state_dim]``.
        P_lin_sqr: Square-root covariance of the linearisation density
            (``P_lin = P_lin_sqr.T @ P_lin_sqr``).
        t: Current time.
        rule: Quadrature rule (see :func:`quadrature_nodes`).
        n_nodes: Nodes per dimension for the Gauss-Hermite rule. Exactness is
            ``2 * n_nodes - 1``, so a degree-``k`` polynomial vector field needs
            ``n_nodes >= (k + 1) / 2`` for ``H`` and ``c`` and
            ``n_nodes >= (2k + 1) / 2`` for ``Omega`` as well (a cubic wants
            ``n_nodes = 4``).

    Returns:
        The :class:`SLRModel` surrogate.
    """
    proj = _arg_projection(measure, m_lin.shape[0])
    dim = proj.shape[0]
    nodes, weights = quadrature_nodes(rule, dim, n_nodes)
    # Any right inverse works: the surrogate is invariant to how the nodes are
    # lifted off the projected subspace, because the residual is affine in the
    # complement and the weighted mean of the lifted points is m_lin.
    lift = np.linalg.pinv(proj)

    # Projected marginal, in square-root form (no noise, no offset). Only its
    # square root is needed: the lifted node is
    # ``m_lin + lift @ (y_i - mu_p)``, so the projected mean cancels.
    _, S_p_sqr = sqr_marginalization(
        proj, np.zeros(dim), np.zeros((dim, dim)), m_lin, P_lin_sqr
    )
    # P_p = S_p_sqr.T @ S_p_sqr, so (nodes @ S_p_sqr) has covariance P_p.
    dev_p = nodes @ S_p_sqr  # [K, p], deviations from the projected mean
    states = m_lin[None, :] + dev_p @ lift.T  # [K, state_dim]

    H_nodes, c_nodes = jax.vmap(lambda x: measure.linearize(x, t=t))(states)
    g_nodes = np.einsum("kij,kj->ki", H_nodes, states) + c_nodes

    # E[jacobian_g] is exactly the SLR matrix (Stein's identity); the affine part
    # of g contributes its constant Jacobian and the nonlinear part contributes
    # E[J_f] composed with the projection.
    H = np.tensordot(weights, H_nodes, axes=1)
    g_bar = weights @ g_nodes
    c = g_bar - H @ m_lin

    # Fit residuals at the nodes. The affine part of g cancels identically here,
    # so this is a purely p-dimensional quantity and sums to zero by weight.
    resid = g_nodes - states @ H.T - c[None, :]
    Omega_sqr = np.sqrt(weights)[:, None] * resid

    # One QR gives the effective noise square root; stacking R_sqr first keeps
    # the result square even when there are fewer nodes than residual rows.
    R_sqr = measure.get_noise(t=t)
    _, R_eff_sqr = np.linalg.qr(np.concatenate([R_sqr, Omega_sqr], axis=0))
    return SLRModel(H=H, c=c, R_eff_sqr=R_eff_sqr, Omega_sqr=Omega_sqr)


def check_arg_projection(
    measure: BaseODEInformation,
    state: ArrayLike,
    *,
    t: ArrayLike = 0.0,
    offsets: ArrayLike | None = None,
    atol: float = 1e-10,
) -> None:
    """Verify that ``measure.E_args`` covers every coordinate ``g`` is nonlinear in.

    An ``E_args`` that is too *small* makes :func:`slr_linearize` silently wrong:
    the quadrature would miss the spread in a direction the vector field
    actually reads. This checks the defining property directly -- the Jacobian of
    ``g`` must be unchanged by any state perturbation that leaves ``E_args @
    state`` fixed -- by perturbing along the null space of ``E_args``.

    Intended for tests and for validating a custom measurement subclass; it is a
    concrete-value check and cannot run inside ``jit``.

    Args:
        measure: Measurement model to check.
        state: Base state to check at, shape ``[state_dim]``.
        t: Current time.
        offsets: Perturbation magnitudes to try along each null-space direction.
            Defaults to ``(1.0, -3.7)``.
        atol: Absolute tolerance on the Jacobian difference.

    Raises:
        ValueError: If some null-space perturbation changes the Jacobian, i.e.
            ``E_args`` omits a coordinate the residual is nonlinear in.
    """
    state = np.asarray(state)
    proj = _arg_projection(measure, state.shape[0])
    offs = (1.0, -3.7) if offsets is None else tuple(onp.asarray(offsets).reshape(-1))

    # Orthonormal basis of null(E_args) from the SVD's trailing right singular
    # vectors: exactly the directions the projection cannot see.
    _, s, vt = onp.linalg.svd(onp.asarray(proj))
    rank = int((s > s[0] * 1e-12).sum()) if s.size else 0
    null_basis = vt[rank:]
    if null_basis.shape[0] == 0:  # projection is the full state; nothing to check
        return

    H_ref = measure.jacobian_g(state, t=t)
    for direction in null_basis:
        for off in offs:
            H_pert = measure.jacobian_g(state + off * np.asarray(direction), t=t)
            err = float(np.max(np.abs(H_pert - H_ref)))
            if err > atol:
                raise ValueError(
                    "E_args does not span the coordinates the residual is "
                    f"nonlinear in: perturbing the state by {off} along a "
                    f"null-space direction of E_args changed the Jacobian of g "
                    f"by {err:.3e} (> atol={atol:.1e}). Override E_args on the "
                    "measurement model to include every coordinate the vector "
                    "field reads."
                )
