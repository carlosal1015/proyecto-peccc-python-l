import numpy as np

def plotea(X, Y, Z):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap=cm.rainbow,
                           linewidth=0, antialiased=False, alpha=0.8)
    ax.view_init(45, 230)
    plt.draw()
    ax.grid()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    # ax.set_zlim3d([1e-10, 4e-8])
    ax.set_title('Función FRO')
    plt.show()
    plt.close()


def fro(X, Y):
    return np.abs((X ** 2 - 1) * (Y ** 2 - 1)) / (1 + X ** 2 + Y ** 2)


# Crear una rejilla. Nota: usar la función ```np.meshgrid``` para ello.
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)

X, Y = np.meshgrid(x, y)

# Obtener una versión vectorizada de la función ```fro```.
fro = np.vectorize(fro)

# Calcular la matriz Z evaluando la rejilla en la función ```fro```.
Z = fro(X, Y)

# Ejecutar la función ```plotea``` con la rejilla y la matriz Z
plotea(X, Y, Z)
