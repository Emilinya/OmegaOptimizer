import subprocess
import time

def run():
    print("Running program:")
    subprocess.run(["./target/release/omega_optimizer"])

def preprocess():
    print("Creating rust functions ...")
    start = time.time()
    subprocess.run(["python3", "preprocessor/process_data.py"])
    print(f"Created rust functions in {(time.time() - start):.2g} s")
    
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

    with open("data/datafile.dat", "r") as infile:
        current_data = infile.readline()

    if prev_data != current_data:
        preprocess()
        with open("make/data_comp.dat", "w") as outfile:
            outfile.write(current_data)
    run()
    