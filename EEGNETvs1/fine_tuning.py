import numpy as np
import pickle
from scipy.spatial import procrustes
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ========== STEP 1: Load Datasets ==========
with open("eeg_fft_dataset.pkl", "rb") as f:
    old_data = pickle.load(f)
X_train = old_data["data"]  # (50, 250, 4, 60)
y_train = old_data["labels"]  # (12500,)
X_train = X_train[:2]
y_train = y_train[:500]

with open("eeg_fft_AMY_dataset.pkl", "rb") as f:
    new_data = pickle.load(f)
X_new = new_data["data"]  # (50, 250, 4, 60)
y_new = new_data["labels"]  # (12500,)

# ========== STEP 2: Flatten FFT data ==========
X_train_flat = X_train.reshape(-1, 4 * 60)  # (12500, 240)
X_new_flat = X_new.reshape(-1, 4 * 60)      # (12500, 240)

# ========== STEP 3: Normalize both datasets BEFORE alignment ==========
mean = np.mean(X_train_flat, axis=0)
std = np.std(X_train_flat, axis=0) + 1e-10  # avoid div zero

X_train_norm = (X_train_flat - mean) / std
X_new_norm = (X_new_flat - mean) / std

# ========== STEP 4: Apply Procrustes Alignment ==========
min_samples = min(X_train_norm.shape[0], X_new_norm.shape[0])
X_train_sub = X_train_norm[:min_samples]
X_new_sub = X_new_norm[:min_samples]

_, X_new_aligned, _ = procrustes(X_train_sub, X_new_sub)

# ========== STEP 5: Reshape and prepare data for model ==========
X_new_aligned_reshaped = X_new_aligned.reshape(-1, 4, 60)
X_train_norm = X_train_norm.reshape(-1, 4, 60)
X_train_norm = X_train_norm[..., np.newaxis]
# Add channel dimension if your model expects it (check model input shape)
X_new_aligned_reshaped = X_new_aligned_reshaped[..., np.newaxis]

# Prepare labels (keep as integers since you use sparse_categorical_crossentropy)
y_new_aligned = y_new[:min_samples]

# ========== STEP 6: Load Pretrained Model ==========
model = load_model("eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5")

# Optional: freeze layers if desired
# for layer in model.layers[:-1]:
#     layer.trainable = False
for layer in model.layers:
    layer.trainable = True

# ========== STEP 7: Compile model ==========
# model.compile(
#     optimizer=Adam(1e-4),
#     loss="sparse_categorical_crossentropy",
#     metrics=["accuracy"]
# )

# ========== STEP 8: Fine-tune ==========
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
X_combined = np.concatenate([X_train_norm, X_new_aligned_reshaped], axis=0)
y_combined = np.concatenate([y_train, y_new[:X_new_aligned_reshaped.shape[0]]], axis=0)

# Shuffle combined dataset
indices = np.arange(len(y_combined))
np.random.shuffle(indices)
X_combined = X_combined[indices]
y_combined = y_combined[indices]

# Fine-tune
# model.fit(
#     X_combined,
#     y_combined,
#     batch_size=32,
#     epochs=25,
#     validation_split=0.2,
#     callbacks=[early_stop]
# )

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# ========== STEP 9: Save fine-tuned model ==========
#model.save("eegnet_fft_finetuned_aligned_2_withoutfrozenfirstlayer_andtwoofolddata.h5")

print("✅ Fine-tuning complete and model saved as eegnet_fft_finetuned_aligned.h5")
# ========== STEP 10: Evaluate model (optional) ==========
model2 = load_model("eegnet_fft_finetuned_aligned_2_withoutfrozenfirstlayer_andoneofolddata.h5")
# Assuming 'model' is your loaded or fine-tuned Keras model

# Example: Evaluate on new aligned data
# loss, accuracy = model2.evaluate(X_new_aligned_reshaped, y_new_aligned, batch_size=32)
# print(f"Evaluation on new aligned data - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# # Optional: Evaluate on old training data to check retention
# loss_old, accuracy_old = model2.evaluate(X_train_norm, y_train, batch_size=32)
# print(f"Evaluation on old training data - Loss: {loss_old:.4f}, Accuracy: {accuracy_old:.4f}")

# print(model.summary())


##############################Visualization of Confusion Matrices##############################
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# # # === Get predictions
y_pred_probs = model2.predict(X_new_aligned_reshaped)

y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.array(y_new_aligned)

print(y_true)

print(y_pred.shape, y_true.shape)

# === Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 Test Accuracy: {acc:.4f}")

# === Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred))

# === Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\n🔀 Confusion Matrix:")
print(cm)

# === Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()