import jax
import matplotlib.pyplot as plt
import numpy as np
from einops import rearrange
from jax import numpy as jnp
from jax.nn import sigmoid
from matplotlib import dates
from matplotlib import pyplot as plt
from scipy.linalg import expm
from tueplots import bundles
from tueplots.constants.color import rgb

#jax.config.update('jax_enable_x64', True)

plt.rcParams.update(bundles.beamer_moml())
plt.rcParams.update({"figure.dpi": 200})

data = jnp.load("data_info.npz")
S = data["sird_data"][:, 0] / 1000
I = data["sird_data"][:, 1] / 1000
R = (data["sird_data"][:, 2] + data["sird_data"][:, 3]) / 1000

data_grid = jnp.array(data["data_grid"], dtype=int)
# print(data["date_range_x"].shape)
data_dates = data["date_range_x"][data_grid]

# Use this Kalman Filter
I_data = I[data_grid]
n = len(I_data)

# some np-fu to get useful plotting grid:
extra_t = np.array(
    [data_dates[-1] + (data_dates[-1] - data_dates[-2]) * i for i in range(3 * 30)]
)
t_plot = np.concatenate([data_dates, extra_t])


def lti_disc(F, L, dt=1):
    """
    %LTI_DISC  Discretize LTI ODE with Gaussian Noise
    %
    % Syntax:
    %   [A,Q] = lti_disc(F,L,Qc,dt)
    %
    % In:
    %   F  - NxN Feedback matrix
    %   L  - NxL Noise effect matrix        (optional, default identity)
    %   dt - Time Step                      (optional, default 1)
    %
    % Out:
    %   A - Transition matrix
    %   Q - Discrete Process Covariance
    %
    % Description:
    %   Discretize LTI ODE with Gaussian Noise. The original
    %   ODE model is in form
    %
    %     dx/dt = F x + L w,  w ~ N(0,I)
    %
    %   Result of discretization is the model
    %
    %     x[k] = A x[k-1] + q, q ~ N(0,Q)
    %
    %   Which can be used for integrating the model
    %   exactly over time steps, which are multiples
    %   of dt.

    % History:
    %   11.01.2003  Covariance propagation by matrix fractions
    %   20.11.2002  The first official version.
    %
    % Copyright (C) 2002, 2003 Simo Särkkä
    %
    % $Id: lti_disc.m 111 2007-09-04 12:09:23Z ssarkka $
    %
    % This software is distributed under the GNU General Public
    % Licence (version 2 or later); please refer to the file
    % Licence.txt, included with the software, for details.
    """
    A = expm(F * dt)

    """
    Closed form integration of covariance
    by matrix fraction decomposition
    """
    n = F.shape[0]
    Phi = jnp.vstack([jnp.hstack([F, L @ L.T]), jnp.hstack([jnp.zeros((n, n)), -F.T])])
    AB = expm(Phi * dt) @ jnp.vstack([jnp.zeros((n, n)), jnp.eye(n)])
    Q = jnp.linalg.solve(AB[n : (2 * n), :].T, AB[0:n, :].T).T

    return A, Q


def plot_matrices():
    fig, axs = plt.subplots(2, 3)
    for i in range(6):
        title = ["F", "L", "A", "Q", "H", "R"][i]
        image = eval(title)
        ax = axs.flat[i]
        im = ax.imshow(image)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.show()


def plot_Ps(Ps, rows=3, columns=6):
    max_idx = Ps.shape[0]
    r, c = rows, columns
    fig, axs = plt.subplots(r, c, figsize=(c * 2, r * 2))
    axs = axs.flatten()

    indices = np.linspace(0, max_idx, num=rows * columns, dtype=int)
    for j, i in enumerate(indices):
        ax = axs[j]
        min_max = max(jnp.max(Ps[i]), -jnp.min(Ps[i]))
        im = ax.imshow(Ps[i], vmin=-min_max, vmax=min_max)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_title("P at Step " + str(indices[j]))
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.show()


def plot_state_raster(
    Ps,
    ms,
    h,
    num_derivatives,
    num_states,
    states=["S", "I", "R"],
):
    fig, axs = plt.subplots(num_derivatives, num_states, sharex=True, squeeze=False)
    n_plot = len(t_plot)

    assert len(states) == num_states

    std_r = jnp.sqrt(
        Ps.reshape(n_plot, num_states, num_derivatives, num_states, num_derivatives)
    )
    # std = jnp.zeros((n_plot, num_states, num_derivatives))
    # for s in range(num_states):
    #     for d in range(num_derivatives):
    #         std = std.at[:,s,d].set(std_r[:, s, d, s, d])
    #
    # std = std.reshape(n_plot, num_states * num_derivatives)
    std = jnp.sqrt(jnp.diagonal(Ps, 0, 1, 2))


    ms_h = jax.vmap(h)(ms)
    lower = jax.vmap(h)(ms - 2 * std)
    upper = jax.vmap(h)(ms + 2 * std)

    ms_r = ms_h.reshape((n_plot, num_states, num_derivatives))
    lower = lower.reshape((n_plot, num_states, num_derivatives))
    upper = upper.reshape((n_plot, num_states, num_derivatives))

    # alim = [(-0.1, 1.1), (-0.02, 0.02), (-0.02, 0.2)]
    # scale = [1, 100, 1000]
    for s, state in enumerate(states):
        for d in range(num_derivatives):
            ax = axs[d, s]
            std = std_r[:, s, d, s, d]
            if state == "I":
                if d == 0:
                    # plot data
                    ax.plot(data_dates, I_data, ".", color=rgb.tue_dark, lw=1, ms=2)

            # plot mean
            ax.plot(t_plot, ms_r[:, s, d], "-", color=rgb.tue_blue, lw=1)

            # calculate_axis
            if np.any(np.isnan(ms_r[:, s, d])):
                # some numerical problems, calculate axis manually for before
                temp = ms_r[:, s, d]
                first_nan_idx = np.where(np.isnan(temp))[0][0]
                end_idx = first_nan_idx * 3 // 4
                temp = temp[0:end_idx]
                ymin, ymax = temp.min(), temp.max()
                if not ymin == ymax:
                    diff = ymax - ymin
                    ax.set_ylim((ymin - 0.05 * diff, ymax + 0.05 * diff))
                    ax.set_ymargin(0.1)

            elif not np.all(ms_r[:, s, d] == 0):
                # only use autoscale on plot of mean, not of uncertainty:
                ax.autoscale_view()  # force auto-scale to update data limits
                # if not np.all(np.isclose(ylim, default) for (ylim, default) in zip(ax.get_ylim(), (-0.055, 0.055))):
                ax.set_autoscale_on(False)

            # plot variation
            ax.fill_between(
                t_plot,
                lower[:,s,d],
                upper[:,s,d],
                color=rgb.tue_blue,
                alpha=0.2,
            )
            # ax.set_ylim(0, 0.01);
            ax.xaxis.set_tick_params(rotation=45)
            ax.axhline(0, color=rgb.tue_dark, lw=0.5)
            if d == 0:
                ax.set_title(r"$" + state + "$")
            if d == 1:
                ax.set_title(r"$\dot{" + state + r"}$")
            if d == 2:
                ax.set_title(r"$\ddot{" + state + r"}$")
            # ax.set_ylim([a / scale[d] for a in alim[s]])
            # ax.set_xlim( (np.float64(18272.15), np.float64(18410.85)))
            ax.grid(axis="x")
    return fig, axs


def build_A_Q(num_states, num_derivatives, sigma):
    assert len(sigma) == num_states

    # empty F, L
    # we stack first along states and derivatives for easier handling
    F = jnp.zeros((num_states, num_derivatives, num_states, num_derivatives))
    L = jnp.zeros((num_states, num_derivatives, num_states, num_derivatives))

    for i in range(num_states):
        # select part of tensor that represents this state
        # and set to triangular matrix as seen above
        F = F.at[i, :, i, :].set(jnp.diag(jnp.ones(num_derivatives - 1), k=1))

        # encode gaussian noise in second derivation
        # li = jnp.zeros((num_derivatives, 1))
        # li = li.at[:, -1].set(sigma[i])
        # L = L.at[i, :, i, :].set(li)
        L = L.at[i, 2, i, 2].set(sigma[i])

    F = rearrange(F, "s1 d1 s2 d2 -> (s1 d1) (s2 d2)")
    L = rearrange(L, "s1 d1 s2 d2 -> (s1 d1) (s2 d2)")

    A, Q = lti_disc(F, L, dt=1)
    return F, L, A, Q
