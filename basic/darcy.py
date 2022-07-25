import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sympy as sy
import numpy as np

from scipy.sparse import spdiags


def Creating_Matrix_A(Nx, Ny, r):
    diag0 = np.tile(np.full(Nx, 1 - 4 * r), Nx)
    diag1 = np.tile(np.concatenate(
        (np.array([0.0, 2 * r]), np.full(Nx - 2, r))), Nx)
    diag_1 = np.flip(diag1)
    diag_sup = np.concatenate((np.zeros(Nx), np.full(
        Nx, 2 * r), np.tile(np.full(Nx, r), Nx - 2)))
    diag_inf = np.flip(diag_sup)

    return spdiags([diag_inf, diag_1, diag0, diag1, diag_sup], [-Nx, -1, 0, 1, Nx], Nx**2, Ny**2).tocsr()


A = Creating_Matrix_A(3, 3, 0.5)

# Imprimiendo matriz de prueba A con Sympy
Ap = sy.Matrix(A.todense())
Ap

with plt.style.context('seaborn-darkgrid'):
    plt.spy(A, markersize=2)
    plt.title(r'Estructura de la matriz $A$')

plt.savefig("sparsity_pattern_A.pdf")
plt.close()


def creating_matrix_R(Nx, Ny, dx, r, D):
    R = np.zeros(Nx*Ny)
    R[2 * Nx - 1: Nx * Ny - 1: Nx] = -2.0 * r * dx * D

    return spdiags([R], [0], Nx**2, Ny**2)


R = creating_matrix_R(3, 3, 0.1, 0.5, 0.2)
sy.Matrix(R.todense())

with plt.style.context('seaborn-darkgrid'):
    plt.spy(R, markersize=2)
    plt.title('Estructura de la matriz R')

plt.savefig("sparsity_pattern_R.pdf")
plt.close()

sy.Matrix((A + R).todense())


def creating_vector_b(Nx, Ny, dx, r, D, g):
    # Crear vector completo de ceros
    b = np.zeros(Nx*Ny)
    # Sustituir a partir del segundo bloque hasta el penúltimo, el último
    # elemento con el valor del flujo de las condiciones de fronteras
    # de Nuemann.
    b[2 * Nx - 1: Nx * Ny - 1: Nx] += 2.0 * r * dx * D * g
    return b


b = creating_vector_b(3, 3, 0.1, 0.5, 0.2, np.full(1, 2))
sy.Matrix(b)


def gfun(t, x):
    if t < 1.0:
        return np.concatenate((np.zeros(Nx // 2 - 4), np.full(6, 1e-4), np.zeros(Nx // 2 - 4)))
    else:
        return np.zeros(Nx - 2)


# Coeficiente de almacenamiento [-]
S = 5e-5
# Espesor   [m]
mu = 1.0
# Conductividad Hidráulica  [m/day] --> [m/s]
K = 1e-1 / (24 * 3600)
# infiltración del flujo de agua en la superficie [m/s]
fg = 1e-4

# Coeficiente de difusión
D = K * mu / S

r = 0.2
dx = 0.1
dy = 0.1

Nx = 50
Ny = 50

dx = 0.1
dy = 0.1

x = np.arange(0, Nx)
y = np.arange(0, Ny)

X, Y = np.meshgrid(x, y)

g = gfun(0.0, x)

u_n = np.zeros((Nx, Ny)).flatten()

dt = r * dx**2 / D  # CFL - Condition

t = 0.0
T = 8000

A = Creating_Matrix_A(Nx, Ny, r)
R = creating_matrix_R(Nx, Ny, dx, r, D)
b = creating_vector_b(Nx, Ny, dx, r, D, g)

for k in range(1, T):
    t += dt
    g = gfun(t, x)
    b = creating_vector_b(Nx, Ny, dx, r, D, g)

    u_n = (A + R) * u_n + b

    Z = u_n.reshape((Nx, Ny))

    if k % 100 == 0:
        cm_size = 1 / 2.54

        fig = plt.figure(tight_layout=True, figsize=(
            40 * cm_size, 23 * cm_size), )
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        surf = ax.plot_surface(X, Y, Z, cmap=cm.gist_earth,
                               linewidth=0, antialiased=False, alpha=0.5)

        ax.view_init(75, 230)
        plt.draw()

        ax.grid()
        ax.set_xlabel('prof')
        ax.set_ylabel('x')
        ax.set_zlim3d([1e-10, 4e-8])

        ax.set_title('Infiltración de agua en grava (2D)\nt={0} min, D={1} $m^2/s$, dx={2} $m$, dt={3} $s$\n'
                     'S={4} [-], K={5} $m/s$, g={6} $m/s$, $\mu=${7} m'
                     .format(np.round(t / 60, 2), np.format_float_scientific(D, 3), dx, np.round(dt, 2),
                             np.format_float_scientific(
                         S, 2), np.format_float_scientific(K, 2),
                         np.format_float_scientific(fg, 2), mu))

        # plt.show()
        plt.savefig('darcy/time_%05d.png' % k)
        plt.close()

    print('time =', t)
