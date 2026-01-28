import subprocess
import matplotlib.pyplot as plt
from pipeline import import_dataset
import shutil

L = min(11, len(import_dataset(filepath="datasets/pos_sequences.fasta")[0]))  
# NOTE: later need to assert that all sequences are of equal lengths

AUC_scores = {}
for i in range(0, L):
    result = subprocess.run(['python', 'pipeline.py', str(i)], capture_output=True, text=True)
    output = result.stdout.strip()
    AUC_scores[i] = output
    print(f"pipeline done for n = {i}")

X = list(AUC_scores.keys())
Y = [float(i) for i in list(AUC_scores.values())]

plt.plot(X, Y, "o:b")
plt.xlabel("Order of the markov model")
plt.xticks(X)
plt.ylabel("AUC-ROC value")
plt.title("AUC_ROC plot for different n-th order markov models")
plt.grid(True)

plt.savefig(f"AUC_ROC_comparison.png", dpi=300)
plt.close()

shutil.rmtree('__pycache__', ignore_errors=True)