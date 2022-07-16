import numpy as np
import matplotlib.pyplot as plt

filename = ""
with open("plotting/data.dat") as file:
    filename = file.readline().replace("\n", "");
x_ray, y_ray, amin_ray = np.loadtxt("plotting/data.dat", skiprows=1).T
plt.plot(x_ray, y_ray, ".", label="data")
plt.plot(x_ray, amin_ray, label="optimal params")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.savefig(filename, dpi=200)