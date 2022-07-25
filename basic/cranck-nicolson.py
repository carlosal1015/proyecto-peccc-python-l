
import sympy as sy
import numpy as np

# Ploteo de soluciones
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Matrices sparse
from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


def create_A(r, N):
    diag_0 = np.full((N * N), 1. + 2. * r)

    diag1 = np.tile(np.concatenate(
        (np.array([0.0]), np.full(N - 1, -r / 2.))), N)
    diag_1 = np.tile(np.concatenate(
        (np.full(N - 1, -r / 2.), np.array([0.0]))), N)

    diag_N = np.full((N * N), -1 * r / 2.)

    return spdiags([diag_N, diag_1, diag_0, diag1, diag_N], [-N, -1, 0, 1, N], N ** 2, N ** 2).tocsr()


A = sy.Matrix(create_A(0.5, 5).todense())
A


def create_B_hat(r, N):
    diag_0 = np.full((N * N), 1. - 2. * r)

    diag1 = np.tile(np.concatenate(
        (np.array([0.0]), np.full((N - 1,), r / 2.))), N)
    diag_1 = np.tile(np.concatenate(
        (np.full((N - 1,), r / 2.), np.array([0.0]))), N)

    diag_N = np.full((N * N), r / 2.)

    return spdiags([diag_N, diag_1, diag_0, diag1, diag_N], [-N, -1, 0, 1, N], N ** 2, N ** 2).tocsr()


B_hat = sy.Matrix(create_B_hat(0.5, 5).todense())
B_hat


def fro(X, Y):
    return np.abs((X ** 2 - 1) * (Y ** 2 - 1)) / (1 + X ** 2 + Y ** 2)


# Vectorizando la función de mcondiciones iniciales
fro = np.vectorize(fro)

# Discretización del dominio

dx = dy = 1 / 5.

x = np.arange(-1, 1 + dx, dx)
y = np.arange(-1, 1 + dx, dy)

Nx = x.shape[0]
Ny = y.shape[0]

XX, YY = np.meshgrid(x, y)

# Coeficiente de estabilización (CFL)
r = 0.25

# Coeficiente de difusión
D = 1e-3

# Calculando dt
dt = r * dx ** 2 / D

# Tiempo inicial
t = 0.0
# Tiempo total
T = 77

# Inicialización de variable para condiciones iniciales.
# Las fronteras serán iguales a cero
u = np.zeros((Nx, Ny))

# Aplicar función de condiciones iniciales a u
u = fro(XX, YY) * 200.

# Crear matriz A
A = create_A(r, u.shape[0] - 2)

# Crear matriz B sombrero
B_hat = create_B_hat(r, u.shape[0] - 2)

# Ciclo principal
for k in range(1, T):
    # Convertir u de 2D a 1D usando función aplanar (flatten)
    u_flat = u[1:-1, 1:-1].flatten()

    # Creando vector de carga (load vector en inglés)
    b = B_hat * u_flat

    # Resolución del sistema de ecuaciones para matrices sparse.
    # Se obtiene la nueva solución en el timepo t + dt * k
    u_flat = spsolve(A, b)

    # Convertir u de 1D a 2D nuevamente
    u[1:-1, 1:-1] = u_flat.reshape((u.shape[0] - 2, u.shape[1] - 2))

    # Avanzar en el tiempo
    t += k * dt

    # Plotear solución y salvarla como png
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cb = ax.contourf(XX, YY, u, cmap=cm.jet,
                     levels=np.linspace(0, 200 + dt, 50), alpha=0.5)
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(cb)
    plt.grid()
    plt.title('t%f' % t)

    plt.savefig('cr/time_%05d.png' % k)
    plt.close()
