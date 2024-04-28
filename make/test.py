import subprocess
import platform
import time
import sys
import os
from os.path import isfile

python = "python" if platform.system() == "Windows" else "python3"
force_compile = "0" if len(sys.argv) < 2 else sys.argv[1]
name = "" if len(sys.argv) < 3 else sys.argv[2]

names = ["sqrt", "line", "normal", "sine"]

for file in os.listdir("figures"):
    if isfile(f"figures/{file}"):
        if file.split("_")[0] in names:
            subprocess.run(["rm", f"figures/{file}"])

if name.strip() == "":
    for name in ["sqrt", "line", "normal", "sine"]:
        print(f"Creating test data {name}")
        start = time.time()
        subprocess.run([python, "data/create_data.py", name])
        print(f"Created test data in {(time.time() - start):.2g} s")
        subprocess.run([python, "make/run.py", "datafile.dat", force_compile, name])
        print()
else:
    print(f"Creating test data {name}")
    start = time.time()
    subprocess.run([python, "data/create_data.py", name])
    print(f"Created test data in {(time.time() - start):.2g} s")
    subprocess.run([python, "make/run.py", "datafile.dat", force_compile, name])
    print()
