import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("eegnet_fft_finetuned_aligned_2_withoutfrozenfirstlayer_andoneofolddata.h5")
import pickle
# Load your datasets
with open("eeg_fft_AMY_dataset.pkl", "rb") as f:
    new_data = pickle.load(f)

with open("eeg_fft_dataset.pkl", "rb") as f:
    old_data = pickle.load(f)

X_new = np.array(new_data["data"])    # shape: (N, 4, 1000)
y_new = np.array(new_data["labels"])

print(X_new.shape, y_new.shape)

X_old = np.array(old_data["data"])    # shape: (N, 4, 1000)
y_old = np.array(old_data["labels"])
X_new = X_new.reshape(-1, 4, 60)
X_old = X_old.reshape(-1, 4, 60)

# === Normalize (per trial)
X_old = (X_old - X_old.mean(axis=2, keepdims=True)) / X_old.std(axis=2, keepdims=True)
# === Reshape for CNN input
X_old = X_old[..., np.newaxis] 

X_new = (X_new - X_new.mean(axis=2, keepdims=True)) / X_new.std(axis=2, keepdims=True)
# === Reshape for CNN input
X_new = X_new[..., np.newaxis] 

# Predict on both datasets
y_pred_new = np.argmax(model.predict(X_new), axis=1)
y_pred_old = np.argmax(model.predict(X_old), axis=1)

# If your y labels are one-hot encoded
if y_new.ndim > 1:
    y_true_new = np.argmax(y_new, axis=1)
else:
    y_true_new = y_new

if y_old.ndim > 1:
    y_true_old = np.argmax(y_old, axis=1)
else:
    y_true_old = y_old

# Compute confusion matrices
cm_new = confusion_matrix(y_true_new, y_pred_new)
cm_old = confusion_matrix(y_true_old, y_pred_old)

# Plot side-by-side confusion matrices
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

disp_new = ConfusionMatrixDisplay(confusion_matrix=cm_new)
disp_new.plot(ax=axs[0], colorbar=False)
axs[0].set_title("New Aligned Data")

disp_old = ConfusionMatrixDisplay(confusion_matrix=cm_old)
disp_old.plot(ax=axs[1], colorbar=False)
axs[1].set_title("Old Training Data")

plt.tight_layout()
plt.show()
