"""
Generalized plotting utilities for ODE filter notebooks.

This module consolidates common plotting functions used across multiple notebooks
to reduce code duplication and improve maintainability.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from typing import Optional, List, Callable, Tuple, Union


def safe_variance_bands(mean: np.ndarray, variance: np.ndarray, n_sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate safe confidence bands from mean and variance arrays.
    
    Args:
        mean: Mean values
        variance: Variance values (will be clipped to non-negative)
        n_sigma: Number of standard deviations for the bands
        
    Returns:
        lower_bound, upper_bound: Confidence band boundaries
    """
    safe_var = np.maximum(variance, 1e-12)  # Avoid numerical issues
    margin = n_sigma * np.sqrt(safe_var)
    return mean - margin, mean + margin


def plot_single_trajectory(
    ts: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    t_fine: Optional[np.ndarray] = None,
    exact_solution: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
    color: str = 'blue',
    label: str = 'estimate',
    show_points: bool = True,
    x_lim: Optional[Tuple[float, float]] = None,
    y_lim: Optional[Tuple[float, float]] = None,
    n_sigma: float = 2.0
) -> plt.Axes:
    """
    Plot a single trajectory with confidence bands.
    
    Args:
        ts: Time points for the estimate
        mean: Mean trajectory
        variance: Variance trajectory
        t_fine: Fine time grid for exact solution (optional)
        exact_solution: Exact solution values (optional)
        ax: Matplotlib axis to plot on (creates new if None)
        color: Color for the estimate
        label: Label for the estimate
        show_points: Whether to show scatter points
        x_lim, y_lim: Axis limits
        n_sigma: Number of standard deviations for confidence bands
        
    Returns:
        The matplotlib axis used for plotting
    """
    if ax is None:
        fig, ax = plt.subplots()
    
    # Plot exact solution if provided
    if exact_solution is not None and t_fine is not None:
        ax.plot(t_fine, exact_solution, linestyle='--', color='black', label='ground truth')
    
    # Plot estimate
    line_style = '.-' if show_points else '-'
    ax.plot(ts, mean, line_style, color=color, label=label)
    
    # Add confidence bands
    lower, upper = safe_variance_bands(mean, variance, n_sigma)
    ax.fill_between(ts, lower, upper, alpha=0.2, color=color)
    
    # Set limits
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)
    
    return ax


def plot_sample_paths(
    ts: np.ndarray,
    samples: np.ndarray,
    component_labels: List[str],
    residual_func: Optional[Callable] = None,
    observations: Optional[np.ndarray] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[float, float] = (8, 10)
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot multiple sample paths from posterior distribution.
    
    Args:
        ts: Time points
        samples: Sample paths [n_samples, n_time, n_components]
        component_labels: Labels for each component
        residual_func: Function to compute residuals from samples
        observations: Observed data for residual plot
        colors: Color list
        figsize: Figure size
        
    Returns:
        fig, axes: The created figure and axes
    """
    n_samples, n_time, n_components = samples.shape
    has_residual = residual_func is not None
    n_subplots = n_components + (1 if has_residual else 0)
    
    fig, axes = plt.subplots(n_subplots, 1, figsize=figsize, sharex=True)
    if n_subplots == 1:
        axes = [axes]
    
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
    
    # Plot component samples
    for i, (ax, label) in enumerate(zip(axes[:n_components], component_labels)):
        for s in range(n_samples):
            alpha = 0.7 if n_samples <= 10 else 0.3
            ax.plot(ts, samples[s, :, i], alpha=alpha, 
                   color=colors[i % len(colors)],
                   label=f'Sample {s+1}' if i == 0 and s < 5 else None)
        
        ax.set_ylabel(label)
        ax.grid(True)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    # Plot residual samples
    if has_residual:
        ax_residual = axes[-1]
        for s in range(n_samples):
            residual_s = residual_func(samples[s])
            alpha = 0.7 if n_samples <= 10 else 0.3
            ax_residual.plot(ts, residual_s, alpha=alpha, color=colors[6 % len(colors)])
        
        if observations is not None:
            # Fix the dimension mismatch issue
            if observations.ndim > 1:
                obs_to_plot = observations[:, 0]  # Take first component if multidimensional
            else:
                obs_to_plot = observations
            
            # Ensure we have the right number of time points for observations
            if len(obs_to_plot) == len(ts) - 1:
                # Observations start from ts[1:]
                ax_residual.scatter(ts[1:], obs_to_plot, label="observed data", 
                                  marker='x', color=colors[2 % len(colors)], zorder=10)
            elif len(obs_to_plot) == len(ts):
                # Observations align with all time points
                ax_residual.scatter(ts, obs_to_plot, label="observed data", 
                                  marker='x', color=colors[2 % len(colors)], zorder=10)
            else:
                # Try to match dimensions as best as possible
                min_len = min(len(ts), len(obs_to_plot))
                ax_residual.scatter(ts[:min_len], obs_to_plot[:min_len], label="observed data", 
                                  marker='x', color=colors[2 % len(colors)], zorder=10)
        
        ax_residual.set_ylabel('residual')
        ax_residual.set_xlabel('Time')
        ax_residual.grid(True)
        ax_residual.legend(loc='upper right')
        ax_residual.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
    
    return fig, axes


def setup_plot_style():
    """Set up consistent plotting style across notebooks."""
    try:
        from tueplots import axes, figsizes
        plt.rcParams.update(axes.lines())
        plt.rcParams.update({"figure.dpi": 600})
        
        figsize_config = figsizes.neurips2021(nrows=1, ncols=1)
        plt.rcParams.update(figsize_config)
    except ImportError:
        # Fallback if tueplots is not available
        plt.rcParams.update({
            'figure.dpi': 600,
            'figure.figsize': (6, 4),
            'axes.linewidth': 0.8,
            'lines.linewidth': 1.2
        })


def get_default_colors():
    """Get the default color cycle."""
    color_cycler = plt.rcParams['axes.prop_cycle']
    return color_cycler.by_key()['color']


def fix_sample_plotting_dimensions(ts, observations, samples, residual_func):
    """
    Helper function to fix dimension mismatches in sample plotting.
    
    Args:
        ts: Time points array
        observations: Observation data (may be 1D or 2D)
        samples: Sample paths
        residual_func: Function to compute residuals
        
    Returns:
        Fixed observations array and residual samples
    """
    # Handle observations
    if observations is not None:
        if observations.ndim > 1:
            obs_fixed = observations[:, 0]  # Take first component
        else:
            obs_fixed = observations
    else:
        obs_fixed = None
    
    # Compute residual samples
    residual_samples = []
    for s in range(samples.shape[0]):
        try:
            residual_s = residual_func(samples[s])
            residual_samples.append(residual_s)
        except Exception as e:
            print(f"Warning: Could not compute residual for sample {s}: {e}")
            residual_samples.append(np.zeros(len(ts)))
    
    return obs_fixed, residual_samples
