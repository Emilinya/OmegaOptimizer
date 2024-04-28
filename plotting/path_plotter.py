import re
import sys

from sympy.parsing.sympy_parser import parse_expr
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


def get_minmax_vals(descents):
    mins = np.min([np.min(path.T, axis=1) for _, path in descents], axis=0)
    maxs = np.max([np.max(path.T, axis=1) for _, path in descents], axis=0)
    return mins, maxs


_, datafile, parameter_path_file, fig_folder, prefix = sys.argv

with open(datafile, "r", encoding="utf8") as file:
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

x_ray, y_ray = np.loadtxt(datafile, skiprows=skip).T


def err(params):
    return np.sum((f(x_ray, *params) - y_ray) ** 2)


descents = []
with open(parameter_path_file, "r", encoding="utf8") as file:
    descent_mode = ""
    path_list = []
    for line in file:
        line = line.replace("\n", "")
        try:
            params = [float(v) for v in line[1:-1].split(", ")]
            path_list.append(params)
        except ValueError:
            if descent_mode != "":
                descents.append((descent_mode, np.array(path_list)))
                path_list = []
            descent_mode = line
    descents.append((descent_mode, np.array(path_list)))

min_vals, max_vals = get_minmax_vals(descents)
ranges = [(maxv - minv) for minv, maxv in zip(min_vals, max_vals)]
num_params = len(min_vals)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
title = f"Parameter optimization for ${func}({variable}; {pretty_params}) = {sp.latex(symbolic_f)}$"

if num_params == 1:
    a_ray = np.linspace(
        min_vals[0] - ranges[0] * 0.5, max_vals[0] + ranges[0] * 0.5, 100
    )
    err_ray = np.array([err([a]) for a in a_ray])
    plt.plot(a_ray, np.log10(err_ray))
    for color, (name, path_ray) in zip(colors[1:], descents):
        path = path_ray.T[0]
        errors = [np.log10(err([a])) for a in path]
        plt.plot(path, errors, ".", color=color, markersize=2, zorder=9)
        plt.plot(path, errors, color=color, label=name, zorder=9)

    plt.legend()
    plt.xlabel(f"${sp.pretty(sym_params[0])}$")
    plt.ylabel(f"$\\log(E({pretty_params}))$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{prefix}_descent_path.png", dpi=200)
    plt.clf()

elif num_params == 2:
    a_grid, b_grid = np.meshgrid(
        *[
            np.linspace(minv - rang * 0.5, maxv + rang * 0.5, 100)
            for minv, maxv, rang in zip(min_vals, max_vals, ranges)
        ]
    )
    z_grid = np.zeros_like(a_grid)
    for i, params in enumerate(zip(a_grid.flat, b_grid.flat)):
        z_grid.flat[i] = err(params)

    plt.contourf(a_grid, b_grid, np.log10(z_grid), cmap="plasma", levels=75)
    plt.colorbar(
        label=f"$\\log(E({sp.latex(sym_params[0])}, {sp.latex(sym_params[1])}))$"
    )
    for color, (name, path_ray) in zip(colors, descents):
        plt.plot(*path_ray.T, ".", color=color, markersize=2, zorder=9)
        plt.plot(*path_ray.T, color=color, label=name, zorder=9)
    plt.legend()
    plt.xlabel(f"${sp.pretty(sym_params[0])}$")
    plt.ylabel(f"${sp.pretty(sym_params[1])}$")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{fig_folder}/{prefix}_descent_path.png", dpi=200)
    plt.clf()
elif num_params == 3:
    pass

i = 0
log_errors = [[np.log10(err(params)) for params in path_ray] for _, path_ray in descents]
for (name, path_ray), log_error in zip(descents, log_errors):
    index_range = range(i, i + len(path_ray))
    plt.plot(index_range, log_error, label=name, zorder=9)
    plt.plot(index_range, log_error, ".k", markersize=1, zorder=9)
    i += len(path_ray) - 1
plt.legend()
plt.xlabel("step count")
plt.ylabel(f"$\\log(E({pretty_params}))$")
plt.title(title)
plt.tight_layout()
plt.savefig(f"{fig_folder}/{prefix}_error_decrease.png", dpi=200)
