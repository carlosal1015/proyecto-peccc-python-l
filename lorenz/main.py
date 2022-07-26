import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Lorenz parameters and initial conditions
sigma, beta, rho = 10, 2.667, 28
x0, y0, z0 = 0, 1, 1.05

# Maximum time point and total number of time points
tmax, n = 100, 10000


def Lorenz(X, t, sigma, beta, rho):
    """The Lorenz equations"""
    x, y, z = X
    dx = -sigma * (x - y)
    dy = rho*x - y - x*z
    dz = -beta*z + x*y
    return (dx, dy, dz)


# Integrate the Lorenz equations on the time grid t.
t = np.linspace(0, tmax, n)
f = odeint(Lorenz, (x0, y0, z0), t, args=(sigma, beta, rho))
x, y, z = f.T

# Plot the Lorenz attractor using a Matplotlib 3D projection.
with plt.style.context('dark_background'):
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(x, y, z, 'b-', lw=0.5)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('y', fontsize=15)
    ax.set_zlabel('z', fontsize=15)
    plt.tick_params(labelsize=15)
    ax.set_title('Lorenz Attractor', fontsize=15)
plt.savefig("lorenz.pdf")
plt.close()
