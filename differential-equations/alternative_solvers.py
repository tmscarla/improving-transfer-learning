import numpy as np
from scipy.integrate import odeint
# from energy_computations import energy

# Use below in the Scipy Solver
def f(u, t, lam=1):
    x, px = u  # unpack current values of u
    derivs = [px, -x - lam * x ** 3]  # list of dy/dt=f functions
    return derivs


# Scipy Solver
def NLosc_solution(t, x_0, px_0, lam=1):
    u_0 = [x_0, px_0]

    # Call the ODE solver
    solPend = odeint(f, u_0, t, args=(lam,))
    x_P = solPend[:, 0]
    px_P = solPend[:, 1]

    return x_P, px_P

# Symplectic Euler
# def symEuler(Ns, x0, px0, t_0, t_max, lam):
#     t_s = np.linspace(t_0, t_max, Ns + 1)
#     x_s = np.zeros(Ns + 1);
#     p_s = np.zeros(Ns + 1)
#     x_s[0], p_s[0] = x0, px0
#     dts = t_max / Ns;
#
#     for n in range(Ns):
#         x_s[n + 1] = x_s[n] + dts * p_s[n]
#         p_s[n + 1] = p_s[n] - dts * (x_s[n + 1] + lam * x_s[n + 1] ** 3)
#
#     E_euler = energy(x_s, p_s, lam=1)
#     return E_euler, x_s, p_s, t_s
