### Interactive plot - CLEAN VERSION - Copy this into a new Jupyter cell ###

import ipywidgets as widgets
from IPython.display import display

# Default values
DEF = {
    'beta_0': 0.22,
    'gamma': 0.06,
    'std_beta_0': 0.01,
    'std_beta_0_prime': 0.01,
    'std_beta_0_prime_prime': 0.01,
    'sigma_x': 0.00001,
    'sigma_beta': 0.0003,
    'std_R_I': 0.0005,
    'std_R_x': 0.000005
}

def run_filter(beta_0, gamma, std_beta_0, std_beta_0_prime, std_beta_0_prime_prime,
               sigma_x, sigma_beta, std_R_I, std_R_x):
    """Run Kalman filter and plot"""
    
    # Define ODE
    Pop = S[0] + I[0] + R[0]
    def vf_l(x):
        return anp.array([-x[3]*x[0]*x[1], x[3]*x[0]*x[1]-gamma*x[1], gamma*x[1], 0])
    
    y0 = anp.array([S[0], I[0], R[0], beta_0])
    d, q = 4, 2
    
    A_f = lambda h: anp.array([[1., h, h**2/2.], [0., 1., h], [0., 0., 1.]])
    Q_f = lambda h: anp.array([[h**5/20., h**4/8., h**3/6.],
                                [h**4/8., h**3/3., h**2/2.],
                                [h**3/6., h**2/2., h]])
    
    Jvf = jacobian(vf_l)
    x1 = vf_l(y0)
    x2 = anp.dot(Jvf(y0), x1)
    mu_0 = anp.concatenate([y0, x1, x2])
    
    Sigma_0 = anp.eye(d*(q+1)) * 0.0
    Sigma_0[3,3] = std_beta_0
    Sigma_0[7,7] = std_beta_0_prime
    Sigma_0[11,11] = std_beta_0_prime_prime
    
    t0, t1 = data_grid[0], data_grid[-1]
    N = data_grid.shape[0] - 1
    h = (t1 - t0) / N
    
    A_h = anp.kron(A_f(h), anp.eye(d))
    Q_h = anp.kron(Q_f(h), anp.diag([sigma_x, sigma_x, sigma_x, sigma_beta])**2)
    b_h = anp.zeros((q+1)*d)
    
    E0 = anp.kron(anp.array([1., 0., 0.]), anp.eye(d))
    E1 = anp.kron(anp.array([0., 1., 0.]), anp.eye(d))
    
    def g_c(X):
        x = anp.dot(E0, X)
        x_dot = anp.dot(E1, X)
        z_ode = x_dot - vf_l(x)
        z_conserved = anp.array([Pop - anp.sum(x[:3])])
        z_observation = anp.array([x[1]])
        return anp.concatenate([z_ode[:3], z_conserved, z_observation])
    
    jac_g = jacobian(g_c)
    z_seq = anp.zeros((N, d+1))
    z_seq[:,-1] = I[1:]
    
    R_h = anp.eye(d+1)
    R_h[-1,-1] = std_R_I
    R_h[:4,:4] *= std_R_x
    
    m_s, P_s, m_p, P_p = compute_kalman_forward_stable(
        mu_0, Sigma_0, A_h, b_h, Q_h, R_h, g_c, jac_g, z_seq, N
    )
    m_s, P_s = compute_kalman_backward(m_s, P_s, m_p, P_p, A_h, N)
    
    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(5, 3.6), sharex=True)
    
    for ax, col, ylab, ylim in [
        (axs[0], 1, "Infected", [-0.0001, 0.005]),
        (axs[1], 3, r'$\beta(t)$', [-0.05, 0.3])
    ]:
        ax.plot(data_dates, m_s[:, col])
        margin = 2 * np.sqrt(P_s[:, col, col])
        ax.fill_between(data_dates, m_s[:, col]-margin, m_s[:, col]+margin, alpha=0.3)
        if col == 1:
            ax.scatter(data_dates, I, color='black', s=2)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_ylim(ylim)
        ax.tick_params(axis='both', which='major', labelsize=7)
        ax.grid(True, which='major', axis='both', linestyle='-', linewidth=0.6, alpha=0.5)
        ax.xaxis.set_minor_locator(dates.WeekdayLocator(interval=2))
        ax.grid(True, which='minor', axis='x', linestyle=':', linewidth=0.4, alpha=0.3)
    
    for ed in political_events:
        axs[0].axvline(ed, color='red', linestyle='--', linewidth=1, alpha=0.7)
        axs[1].axvline(ed, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    axs[0].yaxis.set_major_formatter(
        plt.matplotlib.ticker.FuncFormatter(lambda x, _: f"{x*100:.1f}%")
    )
    axs[1].xaxis.set_tick_params(rotation=90, labelsize=7)
    axs[1].xaxis.set_major_formatter(dates.DateFormatter("%m / %Y"))
    
    plt.tight_layout()
    plt.show()


# Create widgets
st = {'description_width': '80px'}

w1 = widgets.FloatSlider(value=DEF['beta_0'], min=0.05, max=0.5, step=0.01,
                          description='β₀', style=st, continuous_update=False)
w2 = widgets.FloatSlider(value=DEF['gamma'], min=0.01, max=0.2, step=0.01,
                          description='γ', style=st, continuous_update=False)
w3 = widgets.FloatLogSlider(value=DEF['std_beta_0'], min=-4, max=-1, step=0.1,
                             description='σ(β₀)', style=st, continuous_update=False)
w4 = widgets.FloatLogSlider(value=DEF['std_beta_0_prime'], min=-4, max=-1, step=0.1,
                             description="σ(β₀')", style=st, continuous_update=False)
w5 = widgets.FloatLogSlider(value=DEF['std_beta_0_prime_prime'], min=-4, max=-1, step=0.1,
                             description="σ(β₀'')", style=st, continuous_update=False)
w6 = widgets.FloatLogSlider(value=DEF['sigma_x'], min=-7, max=-3, step=0.1,
                             description='σₓ', style=st, continuous_update=False)
w7 = widgets.FloatLogSlider(value=DEF['sigma_beta'], min=-5, max=-2, step=0.1,
                             description='σᵦ', style=st, continuous_update=False)
w8 = widgets.FloatLogSlider(value=DEF['std_R_I'], min=-5, max=-2, step=0.1,
                             description='σ_obs(PDE)', style=st, continuous_update=False)
w9 = widgets.FloatLogSlider(value=DEF['std_R_x'], min=-7, max=-3, step=0.1,
                             description='σ_obs(I)', style=st, continuous_update=False)

# Reset button
def reset_all(b):
    w1.value = DEF['beta_0']
    w2.value = DEF['gamma']
    w3.value = DEF['std_beta_0']
    w4.value = DEF['std_beta_0_prime']
    w5.value = DEF['std_beta_0_prime_prime']
    w6.value = DEF['sigma_x']
    w7.value = DEF['sigma_beta']
    w8.value = DEF['std_R_I']
    w9.value = DEF['std_R_x']

btn = widgets.Button(description='Reset', button_style='info',
                      layout=widgets.Layout(width='100px'))
btn.on_click(reset_all)

# Layout
c1 = widgets.VBox([widgets.HTML("<b>Model</b>"), w1, w2])
c2 = widgets.VBox([widgets.HTML("<b>Init Unc.</b>"), w3, w4, w5])
c3 = widgets.VBox([widgets.HTML("<b>Proc Noise</b>"), w6, w7])
c4 = widgets.VBox([widgets.HTML("<b>Obs Noise</b>"), w8, w9])

ui = widgets.VBox([
    widgets.HBox([c1, c2, c3, c4]),
    widgets.HBox([btn], layout=widgets.Layout(justify_content='center'))
])

out = widgets.interactive_output(run_filter, {
    'beta_0': w1, 'gamma': w2,
    'std_beta_0': w3, 'std_beta_0_prime': w4, 'std_beta_0_prime_prime': w5,
    'sigma_x': w6, 'sigma_beta': w7,
    'std_R_I': w8, 'std_R_x': w9
})

display(ui, out)

