import time
import sys
import os
from os.path import isfile

from make.make import make_and_run
from data.create_data import create_data, Functions


def run_test(function: Functions, force_compile: bool):
    print(f"Creating test data {function.name.lower()}")
    start = time.time()
    create_data(function)
    print(f"Created test data in {(time.time() - start):.2g} s")
    make_and_run("datafile.dat", force_compile, function.name.lower())
    print()


def main():
    force_compile = bool("0" if len(sys.argv) < 2 else sys.argv[1])
    name = "" if len(sys.argv) < 3 else sys.argv[2]

    if name.strip() == "":
        names = ["sqrt", "line", "normal", "sine"]
    else:
        names = [name]

    # parse functions before deleting stuff just in case ...
    functions = [Functions.from_string(name) for name in names]

    for file in os.listdir("figures"):
        if isfile(f"figures/{file}") and file.split("_")[0] in names:
            os.remove(f"figures/{file}")

    for function in functions:
        run_test(function, force_compile)


if __name__ == "__main__":
    main()
