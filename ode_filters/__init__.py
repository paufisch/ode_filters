"""ODE Filters: Kalman filtering and smoothing for differential equations.

This package provides implementations of Extended Kalman Filters (EKF), Kalman
smoothers, and related utilities for inference in ordinary differential equation
(ODE) systems. Includes:

Modules:
    gaussian_inference: Bayesian inference operations (marginalization, inversion).
    sqr_gaussian_inference: Numerically stable square-root (Cholesky) formulations.
    ODE_filters: Extended Kalman Filter implementation with backward sampling.
    helpers: Core Kalman filter and smoother implementations.
    plotting_utils: Visualization utilities for filter results.
"""
