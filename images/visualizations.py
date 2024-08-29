import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Definir la función y sus derivadas parciales
def f(x, y):
    return (x - 3)**2 + (y - 2)**2 + 4

# Derivada parcial de X
def df_dx(x, y):
    return 2 * (x - 3)

# Derivada parcial de Y
def df_dy(x, y):
    return 2 * (y - 2)

def visualizacion_iteraciones_descenso_gradiente():
    # Parámetros del algoritmo de descenso del gradiente
    alpha = 0.1  # Tasa de aprendizaje
    iterations = 10  # Número de iteraciones
    x0, y0 = -1, 7  # Punto inicial

    # Listas para almacenar los puntos a lo largo de las iteraciones
    x_values = [x0]
    y_values = [y0]
    z_values = [f(x0, y0)]

    # Ejecutar el descenso del gradiente
    for _ in range(iterations):
        # Calcular las derivadas parciales
        dfdx = df_dx(x0, y0)
        dfdy = df_dy(y0, y0)
        
        # Actualizar los valores de x e y
        x0 = x0 - alpha * dfdx
        y0 = y0 - alpha * dfdy
        
        # Guardar los nuevos puntos
        x_values.append(x0)
        y_values.append(y0)
        z_values.append(f(x0, y0))

    # Crear una cuadrícula de valores x e y para la superficie
    x = np.linspace(-1, 7, 400)
    y = np.linspace(-1, 7, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Crear la figura
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la superficie
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)

    # Agregar una barra de color
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    # Graficar los puntos del descenso del gradiente
    ax.scatter(x_values, y_values, z_values, color='red', s=50, label='Points obtained with gradient descent', zorder=5)

    # Configurar etiquetas y título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('3D Visual with gradient descent new points')

    # Mostrar la leyenda
    ax.legend()

    # Mostrar la gráfica
    plt.show()

def visualizacion_gradiente_punto_inicial():
    # Crear una cuadrícula de valores x e y
    x = np.linspace(-1, 7, 400)
    y = np.linspace(-1, 7, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Nuevo punto de interés para mejor visualización
    x0, y0 = -1, 7  # Cambiado el punto de interés
    z0 = f(x0, y0)

    # Derivadas parciales en el punto (x0, y0)
    dfdx = df_dx(x0, y0)
    dfdy = df_dy(x0, y0)

    # Crear la figura
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la superficie
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)  # Se añadió alpha para transparencia

    # Agregar una barra de color
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    # Graficar el punto de interés
    ax.scatter(x0, y0, z0, color='red', s=100, label=f'Point ({x0}, {y0}, {z0})', zorder=5)

    # Graficar las rectas tangentes que representan las derivadas parciales
    # Recta en dirección de x
    ax.quiver(x0, y0, z0, 1, 0, dfdx, color='blue', length=8, normalize=True, label=r'Partial derivative $\frac{\partial f}{\partial x}$', zorder=6)

    # Recta en dirección de y
    ax.quiver(x0, y0, z0, 0, 1, dfdy, color='orange', length=8, normalize=True, label=r'Partial derivative $\frac{\partial f}{\partial y}$', zorder=6)

    # Graficar la dirección del gradiente
    ax.quiver(x0, y0, z0, dfdx, dfdy, dfdx**2 + dfdy**2, color='green', length=8, normalize=True, label='Gradient direction', zorder=7)

    # Configurar etiquetas y título
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('3D Visual with partial derivatives and gradient')

    # Mostrar la leyenda
    ax.legend()

    # Mostrar la gráfica
    plt.show()

def visualizacion_funcion_3D():
    # Crear una cuadrícula de valores x e y
    x = np.linspace(-1, 7, 400)
    y = np.linspace(-1, 7, 400)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    # Crear la figura
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Graficar la superficie
    surface = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

    # Agregar una barra de color
    fig.colorbar(surface, ax=ax, shrink=0.5, aspect=5)

    # Configurar etiquetas
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x, y)')

    # Título de la gráfica
    ax.set_title('Visualización 3D de la función f(x, y) = (x - 3)^2 + (y - 2)^2 + 4')

    # Mostrar la gráfica
    plt.show()