import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    sigma = 1000 # Noise sigma
    dt = .001  # Time step.
    T = 10.  # Total time.
    n = int(T / dt)  # Number of time steps.
    timespan = np.linspace(0., T, n)  # Vector of times.

    beta = 0.5
    gamma = 0.5

    sigma_bis = np.sqrt(sigma)
    sqrtdt = np.sqrt(dt)

    # Initialize arrays and force initial conditions
    s = np.zeros(n)
    s[0] = 0.8
    i = np.zeros(n)
    i[0] = 0.2
    r = np.zeros(n)

    N = 1
    for t in range(n - 1):
        noise_term = sigma_bis * sqrtdt * np.random.randn()
        s[t + 1] = s[t] + dt * ((- beta * s[t] * i[t]) / N - noise_term)
        i[t + 1] = i[t] + dt * ((beta * s[t] * i[t]) / N - gamma * i[t] + noise_term)
        r[t + 1] = r[t] + dt * (gamma * i[t])

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(timespan, s, lw=2)
    ax.plot(timespan, i, lw=2)
    ax.plot(timespan, r, lw=2)

    fig.show()
