import subprocess
import time

print("Creating test data")
start = time.time()
subprocess.run(["python3", "data/create_data.py"])
print(f"Created test data in {(time.time() - start):.2g} s")
subprocess.run(["python3", "make/run.py"])