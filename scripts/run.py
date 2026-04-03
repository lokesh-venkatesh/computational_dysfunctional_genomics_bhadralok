import subprocess
import sys
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

all_TFs = ["CTCF"] # ["CTCF", "REST", "EP300"]
chromosome = 4
K = 5 # number of folds for cross-validation
i = 7 # order for which to plot ROC and PRC
orders = range(10, -1, -1)
pseudocounts = 0.1

num_tasks = len(all_TFs)*len(orders) # Determine total number of tasks
MAX_CAP = 6 # max(1, os.cpu_count()-4)  # Compute optimal number of workers
MAX_WORKERS = max(1, min(os.cpu_count(), num_tasks, MAX_CAP))

os.makedirs("outputs", exist_ok=True)

if not os.path.exists("data/dataset.csv"):
    subprocess.run([sys.executable, "2_data_as_one_tsv.py"], check=True)

def run_model(order, transc_factor):
    print(f"Starting TF={transc_factor}, order={order}")
    subprocess.run([sys.executable, "model.py", str(order), str(K),
        str(chromosome), str(transc_factor), str(pseudocounts)], check=True)
    print(f"Finished TF={transc_factor}, order={order}")

for transc_factor in all_TFs:
    """This code block parallelises the pipeline to run for different hyperparams all at once,
    depending on how many processing cores the system can handle at once"""
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(run_model, order, transc_factor) for order in orders]
        for f in as_completed(futures):
            f.result()
shutil.rmtree("__pycache__")