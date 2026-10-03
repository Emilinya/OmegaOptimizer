import os

import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("src/plotting/data.dat", "r", encoding="utf8") as datafile:
        parameters = datafile.readline().strip()

        figure_name = datafile.readline().strip()
        data_length = int(datafile.readline())
        data_in, data_out = np.loadtxt(datafile, max_rows=data_length).T

        output_length = int(datafile.readline())
        x_ray, y_ray = np.loadtxt(datafile, max_rows=output_length).T

    head, _ = os.path.split(figure_name)
    os.makedirs(head, exist_ok=True)

    plt.style.use("dark_background")
    # plt.title(r"$\mathbf{a} = $" + parameters)
    plt.grid(alpha=0.4)
    plt.plot(data_in, data_out, "o", color="#1f77b4", label="input data")
    plt.plot(x_ray, y_ray, color="#ff7f0e", label="best fit model")
    plt.gca().set_facecolor((0, 0, 0, 0.2))
    legend = plt.legend()
    legend.get_frame().set_alpha(None)
    legend.get_frame().set_facecolor((0, 0, 0, 0.2))

    plt.savefig(figure_name, dpi=200, bbox_inches="tight", facecolor=(1,1,1,0))


if __name__ == "__main__":
    main()
