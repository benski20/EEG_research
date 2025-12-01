import os
import numpy as np
import matplotlib.pyplot as plt
import math

folder = "original_model_evaluation_results_corrected"  # Change to your folder path
files = [f for f in os.listdir(folder) if f.endswith("_confusion_matrix.npy")]
files.sort()

num_matrices = len(files)
cols = min(3, num_matrices)   # max 3 per row
rows = math.ceil(num_matrices / cols)

fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))  # smaller figure size

if rows == 1 and cols == 1:
    axes = np.array([[axes]])
elif rows == 1:
    axes = np.array([axes])
elif cols == 1:
    axes = np.array([[ax] for ax in axes])

for idx, fname in enumerate(files):
    r = idx // cols
    c = idx % cols
    ax = axes[r, c]

    cm = np.load(os.path.join(folder, fname))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(fname.replace("_confusion_matrix.npy",""), fontsize=6)  # smaller title
    ax.set_xlabel("Predicted", fontsize=6)
    ax.set_ylabel("True", fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=4)

    # Show numbers in cells with smaller font and better contrast
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{cm[i, j]:.1f}", ha="center", va="center", color=color, fontsize=6)

# Hide any unused subplots
for idx in range(num_matrices, rows*cols):
    r = idx // cols
    c = idx % cols
    fig.delaxes(axes[r, c])

plt.tight_layout()
plt.savefig("plots/original_model_eval_correct.png", dpi=300, bbox_inches='tight')
plt.show()
