import numpy as np
import pickle
from scipy.spatial import procrustes
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ========== STEP 1: Load Datasets ==========
with open("eeg_fft_dataset.pkl", "rb") as f:
    old_data = pickle.load(f)
X_train1 = old_data["data"]  # (50, 250, 4, 60)
y_train1 = old_data["labels"]  # (12500,)
X_train = X_train1[:1]
y_train = y_train1[:250]

with open("eeg_fft_AMY_dataset.pkl", "rb") as f:
    new_data = pickle.load(f)
X_new = new_data["data"]  # (50, 250, 4, 60)
y_new = new_data["labels"]  # (12500,)

# ========== STEP 2: Flatten FFT data ==========
X_train_flat = X_train.reshape(-1, 4, 60)  # (12500, 240)
X_new_flat = X_new.reshape(-1, 4, 60)      # (12500, 240)

# ========== STEP 3: Normalize both datasets BEFORE alignment ==========
X_train_flat= (X_train_flat - X_train_flat.mean(axis=2, keepdims=True)) / X_train_flat.std(axis=2, keepdims=True) # avoid div zero
X_new_flat = (X_new_flat - X_new_flat.mean(axis=2, keepdims=True)) / X_new_flat.std(axis=2, keepdims=True)  # use old data mean/std

# ========== STEP 5: Reshape and prepare data for model ==========
X_train_flat = X_train_flat[..., np.newaxis]
X_new_flat = X_new_flat[..., np.newaxis]  # Add channel dimension if your model expects it (check model input shape)

from tensorflow.keras.utils import to_categorical

# Prepare labels (keep as integers since you use sparse_categorical_crossentropy)
y_train = to_categorical(y_train, num_classes=3)
y_new = to_categorical(y_new, num_classes=3)

# ========== STEP 6: Load Pretrained Model ==========
model = load_model("eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5")

# Optional: freeze layers if desired
for layer in model.layers:
    layer.trainable = True

# ========== STEP 7: Compile model ==========
model.compile(
    optimizer=Adam(1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ========== STEP 8: Fine-tune ==========
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
X_combined = np.concatenate([X_train_flat, X_new_flat], axis=0)
y_combined = np.concatenate([y_train, y_new], axis=0)

print(X_combined.shape, y_combined.shape)
print(X_train_flat.shape, y_train.shape)
print(X_new_flat.shape, y_new.shape)


# Shuffle combined dataset
indices = np.arange(len(y_combined))
np.random.shuffle(indices)
X_combined = X_combined[indices]
y_combined = y_combined[indices]


from sklearn.utils.class_weight import compute_class_weight
import numpy as np
y_labels = np.argmax(y_combined, axis=1)
# Assuming y_combined are integer labels, not one-hot
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labels),
    y=y_labels
)
class_weight_dict = dict(enumerate(class_weights))
class_weight_dict[0] *= 1.75
class_weight_dict[1] *= 1.75

print("Class weights:", class_weight_dict)

# Fine-tune

model.fit(
    X_combined,
    y_combined,
    batch_size=32,
    epochs=25,
    validation_split=0.2,
    callbacks=[early_stop],
    class_weight=class_weight_dict  # Use class weights to handle class imbalance
)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# ========== STEP 9: Save fine-tuned model ==========
model.save("eegnet_fft_finetuned_aligned_2_withoutfrozenfirstlayer_andzerofolddata_andnoalignment_withclassweight.h5")

print("✅ Fine-tuning complete and model saved as eegnet_fft_finetuned_aligned.h5")
# ========== STEP 10: Evaluate model (optional) ==========
model2 = load_model("eegnet_fft_finetuned_aligned_2_withoutfrozenfirstlayer_andzerofolddata_andnoalignment_withclassweight.h5")
# Assuming 'model' is your loaded or fine-tuned Keras model

# Example: Evaluate on new aligned data
loss, accuracy = model2.evaluate(X_new_flat, y_new, batch_size=32)
print(f"Evaluation on new aligned data - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Optional: Evaluate on old training data to check retention  # (12500, 240)
loss_old, accuracy_old = model2.evaluate(X_train_flat, y_train, batch_size=32)
print(f"Evaluation on old training data - Loss: {loss_old:.4f}, Accuracy: {accuracy_old:.4f}")

print(model.summary())

import random
# Select a random trial from the test set
random_index = random.randint(0, len(X_new_flat) - 1)
print(f"Selected Trial Index: {random_index}")
new_trial = X_new_flat[random_index]
new_trial_label = y_new[random_index]
new_trial_label = np.argmax(new_trial_label)  # Convert one-hot to class index
print(f"Trial Label: {new_trial_label}")  # Print the label for the trial


new_trial = new_trial[np.newaxis, ..., np.newaxis]  # shape: (1, 4, 1000, 1)
 # Print the one-hot encoded label for the trial

prediction = model.predict(new_trial)
predicted_class = np.argmax(prediction)

print(f"Predicted Class: {predicted_class}")

# ##############################Visualization of Confusion Matrices##############################
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# # # === Get predictions
y_pred_probs = model2.predict(X_new_flat)

y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.array(y_new)


y_true = np.argmax(y_new, axis=1)
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