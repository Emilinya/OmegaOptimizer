from dataclasses import dataclass
from typing import Any, Callable
from enum import Enum, auto

import sympy as sp
import numpy as np
import numpy.typing as npt


class Functions(Enum):
    Sqrt = auto()
    Line = auto()
    Normal = auto()
    Sine = auto()

    @classmethod
    def from_string(cls, name: str):
        names = list(map(lambda c: c.name, cls))
        try:
            return list(cls)[names.index(name.lower().capitalize())]
        except ValueError as exc:
            raise ValueError(f"Unknown function name '{name}'") from exc


@dataclass
class FunctionData:
    f: Any
    x_ray: npt.NDArray[np.float64]
    variable: str
    parameters: str
    symbolic_f: Any
    true_params: npt.NDArray[np.float64]


def create_function_data(
    symbols: str,
    symbolic_f_generator: Callable,
    true_params: npt.NDArray[np.float64],
    x_min: float,
    x_max: float,
) -> FunctionData:
    variable, parameters = symbols[0], ", ".join(symbols.split(", ")[1:])
    params = sp.symbols(symbols)

    symbolic_f = symbolic_f_generator(*params)
    f = sp.lambdify(params, symbolic_f)

    x_ray = np.linspace(x_min, x_max, 500)

    return FunctionData(f, x_ray, variable, parameters, symbolic_f, true_params)


def create_data(function: Functions):
    np.random.seed(13)
    if function == Functions.Sqrt:
        data = create_function_data(
            "x, a",
            lambda x, a: sp.sqrt(a * x),
            np.random.uniform(0.5, 1.5, 1),
            0,
            10,
        )
    elif function == Functions.Line:
        data = create_function_data(
            "x, a, b",
            lambda x, a, b: a * x + b,
            np.random.uniform(-2, 3, 2),
            -3,
            3,
        )
    elif function == Functions.Normal:
        ps = np.random.uniform(1, 7, 3)
        data = create_function_data(
            "x, A, x_0, sigma",
            lambda x, A, x_0, sigma: A * sp.exp(-(((x - x_0) / sigma) ** 2)),
            ps,
            ps[1] - 3 * ps[2],
            ps[1] + 3 * ps[2],
        )
    elif function == Functions.Sine:
        ps = np.random.uniform(np.pi, np.pi**3, 4)
        T = 2 * np.pi / ps[1]
        data = create_function_data(
            "t, A, omega, phi, b",
            lambda t, A, omega, phi, b: A * sp.sin(omega * t + phi) + b,
            ps,
            -2 * T,
            2 * T,
        )
    else:
        raise ValueError(f"Unknown argument {function}")

    y_ray = data.f(data.x_ray, *data.true_params) * np.random.normal(
        1, 0.1, data.x_ray.shape
    )

    print(
        f"Creating data with f({data.variable}; {data.parameters}) = {str(data.symbolic_f)}"
    )
    print(f"[{data.parameters}] = {data.true_params}")
    with open("data/datafile.dat", "w", encoding="utf8") as datafile:
        datafile.write(
            f"f({data.variable}; {data.parameters}) = {str(data.symbolic_f)}\n"
        )
        for x, y in zip(data.x_ray, y_ray):
            datafile.write(f"{x} {y}\n")
