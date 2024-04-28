import numpy as np
import sympy as sp
import sys

if len(sys.argv) == 1:
    print("create_data must get a function as an argument")
    exit()

np.random.seed(13)
if sys.argv[1] == "sqrt":
    symbols = "x, a"
    variable, parameters = symbols[0], ", ".join(symbols.split(", ")[1:])
    params = sp.symbols(symbols)
    x, a = params

    symbolic_f = sp.sqrt(a*x)
    true_params = [np.random.uniform(0.5, 1.5)]

    f = sp.lambdify(params, symbolic_f)

    x_ray = np.linspace(0, 10, 500)
elif sys.argv[1] == "line":
    symbols = "x, a, b"
    variable, parameters = symbols[0], ", ".join(symbols.split(", ")[1:])
    params = sp.symbols(symbols)
    x, a, b = params

    symbolic_f = a*x + b
    true_params = np.random.uniform(-2, 3, 2)

    f = sp.lambdify(params, symbolic_f)

    x_ray = np.linspace(-3, 3, 500)
elif sys.argv[1] == "normal":
    symbols = "x, A, x_0, sigma"
    variable, parameters = symbols[0], ", ".join(symbols.split(", ")[1:])
    params = sp.symbols(symbols)
    x, A, x0, sigma = params

    symbolic_f = A*sp.exp(-((x - x0)/sigma)**2)
    true_params = np.random.uniform(1, 7, 3)

    f = sp.lambdify(params, symbolic_f)

    x_ray = np.linspace(true_params[1]-3*true_params[2], true_params[1]+3*true_params[2], 500)
elif sys.argv[1] == "sine":
    symbols = "t, A, omega, phi, b"
    variable, parameters = symbols[0], ", ".join(symbols.split(", ")[1:])
    params = sp.symbols(symbols)
    t, A, omega, phi, b = params

    symbolic_f = A*sp.sin(omega*t + phi) + b
    true_params = np.random.uniform(np.pi, np.pi**3, 4)
    T = 2*np.pi / true_params[1]

    f = sp.lambdify(params, symbolic_f)

    x_ray = np.linspace(-2*T, 2*T, 500)
else:
    print(f"Unknown argument {sys.argv[1]}")
    exit()

y_ray = f(x_ray, *true_params) * np.random.normal(1, 0.1, x_ray.shape)

print(f"Creating data with f({variable}; {parameters}) = {str(symbolic_f)}")
print(f"[{parameters}] = {true_params}")
with open("data/datafile.dat", "w") as datafile:
    datafile.write(f"f({variable}; {parameters}) = {str(symbolic_f)}\n")
    for x, y in zip(x_ray, y_ray):
        datafile.write(f"{x} {y}\n")