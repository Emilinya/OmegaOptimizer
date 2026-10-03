from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import numpy.typing as npt


def add_noise(array: npt.NDArray, relative_scale: float):
    minv, maxv = np.min(array), np.max(array)
    return array + np.random.normal(0, (maxv - minv) * relative_scale, array.shape)


class Function(Enum):
    Sqrt = auto()
    Line = auto()
    Normal = auto()
    Sine = auto()

    def get_f(self):
        match self:
            case Function.Sqrt:
                return lambda x, a, b, c, d: a * np.sqrt(b * x + c) + d
            case Function.Line:
                return lambda x, a, b: a * x + b
            case Function.Normal:
                return lambda x, A, x_0, sigma: A * np.exp(-0.5*(((x - x_0) / sigma) ** 2))
            case Function.Sine:
                return lambda t, A, omega, phi, b: A * np.sin(omega * t + phi) + b
            case _:
                raise ValueError(f"??? {self}")

    @classmethod
    def from_string(cls, name: str):
        names = list(map(lambda c: c.name, cls))
        try:
            return list(cls)[names.index(name.lower().capitalize())]
        except ValueError as exc:
            raise ValueError(f"Unknown function name '{name}'") from exc


@dataclass
class FunctionData:
    x_ray: npt.NDArray[np.float64]
    y_ray: npt.NDArray[np.float64]


def create_data(function: Function):
    np.random.seed(13)
    match function:
        case Function.Sqrt:
            true_parameters = np.random.uniform(0.5, 1.5, 4)
            bounds = (0.0, 10.0)
        case Function.Line:
            true_parameters = np.random.uniform(-2, 3, 2)
            bounds = (-3.0, 3.0)
        case Function.Normal:
            true_parameters = np.random.uniform(1, 7, 3)
            x0, std = float(true_parameters[1]), float(true_parameters[2])
            bounds = (x0 - 3 * std, x0 + 3 * std)
        case Function.Sine:
            true_parameters = np.random.uniform(np.pi, np.pi**3, 4)
            T = 2 * np.pi / float(true_parameters[1])
            bounds = (-2 * T, 2 * T)
        case _:
            raise ValueError(f"Unknown function: {function}")

    print(
        f"Creating {function.name.lower()} data with true parameters {true_parameters}"
    )
    x_ray = np.linspace(*bounds, 100)
    y_ray = function.get_f()(x_ray, *true_parameters)
    noise_scale = 0.02

    with open(
        f"data/{function.name.lower()}_data.dat", "w", encoding="utf8"
    ) as datafile:
        for x, y in zip(add_noise(x_ray, noise_scale), add_noise(y_ray, noise_scale)):
            datafile.write(f"{x} {y}\n")


def main():
    for function in [Function.from_string(n) for n in Function._member_names_]:
        create_data(function)


if __name__ == "__main__":
    main()
