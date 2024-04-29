import subprocess
import platform
import time

from preprocessor.process_data import create_function_file


def run(datafile: str, name: str):
    print("Running program:")
    if platform.system() != "Windows":
        subprocess.run(
            [
                "./target/release/omega_optimizer",
                f"datafile={datafile}",
                f"name={name}",
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                "target/release/omega_optimizer.exe",
                f"datafile={datafile}",
                f"name={name}",
            ],
            check=True,
        )


def preprocess(datafile: str):
    with open("preprocessor/initial_params.dat", "w", encoding="utf8") as paramfile:
        paramfile.write("")

    print("Creating rust functions ...")
    start = time.time()
    create_function_file(datafile)
    print(f"Created rust functions in {(time.time() - start):.2g} s")


def compile_rust():
    print("Building program ...")
    start = time.time()
    subprocess.run(["cargo", "build", "--release"], check=True)
    print(f"Built program in {(time.time() - start):.2g} s")


def make_and_run(datafile="datafile.dat", force_compile=False, name=""):
    try:
        with open("make/data_comp.dat", "r", encoding="utf8") as infile:
            prev_data = infile.read()
    except FileNotFoundError:
        prev_data = ""

    with open(f"data/{datafile}", "r", encoding="utf8") as infile:
        current_data = infile.readline()

    if prev_data != current_data:
        preprocess(datafile)
        with open("make/data_comp.dat", "w", encoding="utf8") as outfile:
            outfile.write(current_data)

    if force_compile or prev_data != current_data:
        compile_rust()

    run(datafile, name)
