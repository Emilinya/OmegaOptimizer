from sympy.parsing.sympy_parser import parse_expr
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import re

with open("data/datafile.dat", "r", encoding="utf8") as file:
    match = re.search(r"(.*?)\((.*?);\s*?(.*?)\)\s*?=\s*?(.*)", file.readline())
    if match is None:
        raise ValueError("Waaaa!")

    func, variable, parameters, expression = match[1], match[2], match[3], match[4]

    try:
        x, y = [float(v) for v in file.readline().split()]
        skip = 1
    except ValueError:
        skip = 2

parameters = parameters.replace(" ", "").split(",")
x, *sym_params = sp.symbols(f"{variable}, {' '.join(parameters)}")
pretty_params = ", ".join(sp.latex(sym_param) for sym_param in sym_params)

local_dict = {s: v for s, v in zip(parameters, sym_params)}
local_dict[variable] = x

symbolic_f = parse_expr(expression, local_dict=local_dict)
f = sp.lambdify([x] + sym_params, symbolic_f)


def err(params):
    return np.sum((f(x_ray, *params) - y_ray) ** 2)


x_ray, y_ray = np.loadtxt("data/datafile.dat", skiprows=skip).T


def err(params):
    return np.sum((f(x_ray, *params) - y_ray) ** 2)


minx, maxx, spanx = np.min(x_ray), np.max(x_ray), np.max(x_ray) - np.min(x_ray)
miny, maxy, spany = np.min(y_ray), np.max(y_ray), np.max(y_ray) - np.min(y_ray)

descents = []
with open("plotting/descent_path.dat", "r", encoding="utf8") as file:
    descent_mode = ""
    path_list = []
    for line in file:
        line = line.replace("\n", "")
        try:
            params = [float(v) for v in line[1:-1].split(", ")]
            descents.append(params)
        except ValueError:
            pass
err_xs, err_ys = [0], [np.log(err(descents[0]))]
min_err, max_err = np.min([np.log(err(descent)) for descent in descents]), np.max(
    [np.log(err(descent)) for descent in descents]
)
err_range = max_err - min_err

fig, axs = plt.subplots(2, dpi=200, figsize=(8, 10))

axs[0].plot(x_ray, y_ray, ".")
(ln1,) = axs[0].plot(x_ray, f(x_ray, *descents[0]))
(ln2,) = axs[1].plot(err_xs, err_ys, ".-")


def init():
    axs[0].set_xlim(minx - 0.1 * spanx, maxx + 0.1 * spanx)
    axs[0].set_ylim(miny - 0.1 * spany, maxy + 0.1 * spany)
    axs[1].set_xlim(-0.1, len(descents) - 1 + 0.1)
    axs[1].set_ylim(min_err - 0.1 * err_range, max_err + 0.1 * err_range)

    return [ln1, ln2]


def update(frame):
    print(f"\r{frame+1}/{frames}", end="")
    if frame == frames - 1:
        print()

    frame = frame + 1 - padd_start
    if (frame > 0) and (frame < len(descents) - 1):
        ln1.set_ydata(f(x_ray, *descents[frame]))

        err_xs.append(frame)
        err_ys.append(np.log(err(descents[frame])))
        ln2.set_data(err_xs, err_ys)
    return [ln1, ln2]


padd_start = 2
padd_end = 5
frames = len(descents) - 1 + padd_start + padd_end
ani = FuncAnimation(fig, update, init_func=init, frames=frames, interval=400, blit=True)
ani.save("figures/animation.mp4")
