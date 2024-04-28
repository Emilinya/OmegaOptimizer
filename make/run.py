import subprocess
import platform
import time
import sys

python = "python" if platform.system() == "Windows" else "python3"
datafile = "datafile.dat" if len(sys.argv) < 2 else sys.argv[1]
force_compile = False if len(sys.argv) < 3 else bool(int(sys.argv[2]))
name = "" if len(sys.argv) < 4 else sys.argv[3]

def run():
	print("Running program:")
	if platform.system() != "Windows":
		subprocess.run(["./target/release/omega_optimizer", f"datafile={datafile}", f"name={name}"])
	else:
		subprocess.run(["target/release/omega_optimizer.exe", f"datafile={datafile}", f"name={name}"])

def preprocess():
	with open("preprocessor/initial_params.dat", "w") as paramfile:
		paramfile.write("")

	print("Creating rust functions ...")
	start = time.time()
	subprocess.run([python, "preprocessor/process_data.py", datafile])
	print(f"Created rust functions in {(time.time() - start):.2g} s")

def compile():
	print("Building program ...")
	start = time.time()
	subprocess.run(["cargo", "build", "--release"])
	print(f"Built program in {(time.time() - start):.2g} s")

if __name__ == "__main__":
	try:
		with open("make/data_comp.dat", "r") as infile:
			prev_data = infile.read()
	except FileNotFoundError as e:
		prev_data = ""

	with open(f"data/{datafile}", "r") as infile:
		current_data = infile.readline()

	if prev_data != current_data:
		preprocess()
		with open("make/data_comp.dat", "w") as outfile:
			outfile.write(current_data)

	if force_compile or prev_data != current_data:
		compile()
		
	run()
