from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(int(datetime.now().timestamp()))

inputs = np.random.uniform(-3, 3, size=(30, 2))
weights = np.array([[2, 3], [6, 7]])
b = -22


def neuron(val_inputs, val_weights, b):
    f = (weights[0][0] + weights[0][1]) * val_inputs[0] + (weights[1][0] + weights[1][1]) * val_inputs[1] - b

    if f > 0:
        return 1
    elif f < 0:
        return -1
    else:
        return 0


def plot_perceptron(val_inputs, val_weights, b, resolution=1000):
    x1_values = np.linspace(-5, 5, resolution)
    x2_values = np.linspace(-5, 5, resolution)

    xx, yy = np.meshgrid(x1_values, x2_values)
    zone_points = np.zeros((resolution, resolution))

    x_values = np.linspace(-5, 5, 1000)
    y_values = (b - (weights[0][0] + weights[0][1]) * x_values) / (weights[1][0] + weights[1][1])

    plt.plot(x_values, y_values, label=f'f', color='red')

    for i in range(resolution):
        for j in range(resolution):
            zone_points[i, j] = neuron(np.array([[xx[i, j], yy[i, j]]]).transpose(), val_weights, b)

    plt.contourf(xx, yy, zone_points, levels=[-1, 0, 1], colors=['white', 'grey'], alpha=0.5)

    for input_point in val_inputs:

        output = neuron(input_point, val_weights, b)

        color = 'green' if output > 0 else 'yellow'
        plt.plot(input_point[0], input_point[1], marker='o', markersize=5, color=color)

    plt.xlabel('Input 1')
    plt.ylabel('Input 2')
    plt.title('Perceptron Decision Boundary with Inputs')
    plt.colorbar()
    plt.show()


print(f"equation of the line of the neuron is: {weights[0][0] + weights[0][1]}p1 + {weights[1][0] + weights[1][1]}p2 + "
      f"{b} = 0")

plot_perceptron(inputs, weights, b)
