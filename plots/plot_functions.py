import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from functions.griewank import griewank
from functions.ackley import ackley


def plot_griewank():
    """
    Plot 2D contour and 3D surface for the Griewank function.
    """
    x = np.linspace(-600, 600, 400)
    y = np.linspace(-600, 600, 400)
    X, Y = np.meshgrid(x, y)
    Z = griewank(X, Y)

    # 2D Contour
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cp)
    plt.title('Griewank Function - Contour')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('plots/griewank_contour.png')
    plt.close()

    # 3D Surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    fig.colorbar(surf)
    ax.set_title('Griewank Function - Surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.savefig('plots/griewank_surface.png')
    plt.close()


def plot_ackley():
    """
    Plot 2D contour and 3D surface for the Ackley function.
    """
    x = np.linspace(-32.768, 32.768, 400)
    y = np.linspace(-32.768, 32.768, 400)
    X, Y = np.meshgrid(x, y)
    Z = ackley(X, Y)

    # 2D Contour
    plt.figure(figsize=(6, 5))
    cp = plt.contourf(X, Y, Z, levels=50, cmap='plasma')
    plt.colorbar(cp)
    plt.title('Ackley Function - Contour')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('plots/ackley_contour.png')
    plt.close()

    # 3D Surface
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='none')
    fig.colorbar(surf)
    ax.set_title('Ackley Function - Surface')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    plt.savefig('plots/ackley_surface.png')
    plt.close()
