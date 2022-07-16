import numpy as np
import sympy as sp

symbols = "t, A, omega, phi, b"
variable, parameters = symbols[0], ", ".join(symbols.split(", ")[1:])
params = sp.symbols(symbols)
x, a1, a2, a3, a4 = params

symbolic_f = a1*sp.sin(a2*x + a3) + a4
true_params = [2, 1, np.pi/4, 0.5]

f = sp.lambdify(params, symbolic_f)

x_ray = np.linspace(-np.pi, np.pi, 500)
y_ray = f(x_ray, *true_params) * np.random.normal(1, 0.1, x_ray.shape)

print(f"Creating data with f({variable}; {parameters}) = {str(symbolic_f)}")
print(f"[{parameters}] = {true_params}")
with open("data/datafile.dat", "w") as datafile:
    datafile.write(f"f({variable}; {parameters}) = {str(symbolic_f)}\n")
    for x, y in zip(x_ray, y_ray):
        datafile.write(f"{x} {y}\n")