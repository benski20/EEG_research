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
X_train = X_train1
y_train = y_train1

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
y_labels = np.argmax(y_new, axis=1)
# Assuming y_combined are integer labels, not one-hot
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_labels),
    y=y_labels
)
class_weight_dict = dict(enumerate(class_weights))
class_weight_dict[0] *= 1.25
class_weight_dict[1] *= 1.25

print("Class weights:", class_weight_dict)

# Fine-tune

# model.fit(
#     X_combined,
#     y_combined,
#     batch_size=32,
#     epochs=35,
#     validation_split=0.2,
#     callbacks=[early_stop],
#     class_weight=class_weight_dict  # Use class weights to handle class imbalance
# )

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# ========== STEP 9: Save fine-tuned model ==========
#model.save("eegnet_fft_finetuned_aligned_2_withoutfrozenfirstlayer_and10ofolddata_andnoalignment_withclassweight[1.25:1.25:1].h5")

print("✅ Fine-tuning complete and model saved as eegnet_fft_finetuned_aligned.h5")
# ========== STEP 10: Evaluate model (optional) ==========
model2 = load_model("eegnet_fft_finetuned_aligned_2_withoutfrozenfirstlayer_and10ofolddata_andnoalignment_withclassweight[1.25:1.25:1]_withaugmentation.h5")
# Assuming 'model' is your loaded or fine-tuned Keras model

# # Example: Evaluate on new aligned data
# loss, accuracy = model2.evaluate(X_new_flat, y_new, batch_size=8)
# print(f"Evaluation on new aligned data - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# # Optional: Evaluate on old training data to check retention  # (12500, 240)
# loss_old, accuracy_old = model2.evaluate(X_train_flat, y_train, batch_size=8)
# print(f"Evaluation on old training data - Loss: {loss_old:.4f}, Accuracy: {accuracy_old:.4f}")

# print(model.summary())

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

############################## LIME Analysis ############################
# from lime import lime_tabular
# print(X_combined.shape)
# X_combined = X_combined.reshape(15000, 4 * 60)  # Reshape to (N, 240) for LIME
# print(X_combined.shape)

# # === Define LIME prediction wrapper
# def predict_fn(input_2d):
#     # Reshape back to (N, 4, 1000, 1)
#     #assert input_2d.shape[1] == 240, "Input shape must be (N, 240) for 4 channels and 60 time points"
#     reshaped = input_2d.reshape((-1, 4, 60, 1))
#     print(reshaped.shape)
#     return model2.predict(reshaped)

# # === Define feature names (optional but helpful for interpretability)
# feature_names = [f"ch{ch}_t{t}" for ch in range(4) for t in range(60)]

# # === Initialize LIME Tabular Explainer
# explainer = lime_tabular.LimeTabularExplainer(
#     training_data=X_combined,
#     feature_names=feature_names,
#     class_names=["Class 0", "Class 1", "Class 2"],
#     discretize_continuous=True,
#     mode='classification'
# )

# # === Pick one test sample to explain
# sample_index = 3
# sample = X_combined[sample_index]

# # === Run explanation
# explanation = explainer.explain_instance(
#     data_row=sample,
#     predict_fn=predict_fn,
#     num_features=50  # Top 50 contributing features
# )

# print(f"\n🔍 Explanation for Sample {explanation}:")

# # === Optionally, print top features to console
# print("\n🔍 Top contributing features:")
# for feature, weight in explanation.as_list():
#     print(f"{feature}: {weight:.4f}")



import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# # === Step 1: Raw Input (Feature : Importance)
raw_input = """
ch1_t9 > 1.79: -0.0449
ch1_t10 > 1.45: -0.0444
ch1_t8 > 1.96: 0.0418
ch1_t11 > 1.29: 0.0325
ch1_t7 > 1.78: -0.0298
ch0_t7 > 2.64: 0.0255
ch1_t6 > 1.51: -0.0251
ch0_t8 > 2.75: 0.0235
ch1_t51 <= -0.41: -0.0231
1.23 < ch0_t11 <= 1.76: 0.0215
ch1_t47 <= -0.63: -0.0209
ch1_t33 <= -0.41: 0.0172
1.89 < ch0_t10 <= 2.14: 0.0155
-0.30 < ch2_t31 <= -0.23: -0.0143
ch3_t26 <= 0.41: 0.0141
ch2_t26 > 5.10: 0.0131
ch1_t36 <= -0.46: -0.0131
ch1_t43 <= -0.59: 0.0125
2.37 < ch0_t9 <= 2.58: 0.0115
ch3_t8 > 2.73: -0.0113
0.21 < ch3_t25 <= 1.33: 0.0112
0.19 < ch0_t26 <= 1.00: -0.0110
-1.28 < ch3_t2 <= -1.08: 0.0107
-0.93 < ch3_t3 <= -0.76: -0.0104
ch1_t35 <= -0.46: -0.0102
-0.48 < ch0_t37 <= -0.41: -0.0102
ch2_t25 > 3.85: -0.0101
ch2_t59 > -0.57: -0.0101
ch3_t27 <= -0.18: 0.0100
-0.71 < ch0_t48 <= -0.65: 0.0097
ch2_t10 <= 0.35: -0.0096
ch0_t28 <= -0.21: 0.0095
1.24 < ch3_t11 <= 1.75: -0.0095
0.06 < ch1_t18 <= 0.43: 0.0093
-0.82 < ch0_t54 <= -0.75: -0.0093
ch0_t43 <= -0.61: 0.0090
ch1_t39 <= -0.53: 0.0090
0.42 < ch3_t16 <= 0.76: 0.0088
ch3_t51 <= -0.61: -0.0087
-0.13 < ch1_t27 <= 0.46: -0.0086
ch2_t57 > -0.54: 0.0086
0.33 < ch1_t13 <= 0.98: 0.0084
-0.40 < ch3_t33 <= -0.32: 0.0083
-0.54 < ch0_t40 <= -0.45: -0.0082
-0.73 < ch1_t55 <= -0.51: -0.0082
-1.34 < ch1_t1 <= -0.85: -0.0077
-0.38 < ch3_t32 <= -0.30: 0.0076
ch1_t45 <= -0.61: -0.0074
-0.81 < ch0_t53 <= -0.74: 0.0073
0.65 < ch3_t15 <= 0.97: -0.0069
"""

#=== Step 2: Extract feature -> importance
import re
pattern = re.compile(r"(ch\d+_t\d+)[^:]*:\s*(-?\d+\.\d+)")
matches = pattern.findall(raw_input)

features = [m[0] for m in matches]
importances = [float(m[1]) for m in matches]

# === Step 3: Average contribution per channel
channel_sums = defaultdict(list)
for feat, imp in zip(features, importances):
    ch = feat.split("_")[0]  # ch0, ch1, etc.
    channel_sums[ch].append(imp)

channel_avg = {ch: np.mean(vals) for ch, vals in channel_sums.items()}

# === Step 4: Plot
plt.figure(figsize=(9, 5))
channels = sorted(channel_avg.keys(), key=lambda x: int(x[2:]))
avg_vals = [channel_avg[ch] for ch in channels]

plt.bar(channels, avg_vals, color='skyblue')
plt.title("Average LIME Contribution per Channel For Augmented Generalized Model")
plt.xlabel("Channel")
plt.ylabel("Average Contribution")
plt.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig("plots/eegnet_fft_finetuned_aligned_2_withoutfrozenfirstlayer_and10ofolddata_andnoalignment_withclassweight[1.25:1.25:1]_withaugmentation_featureimportance.png")
plt.show()

# depthwise_layer = model.get_layer("depthwise_conv2d")  # Adjust name if necessary
# print(f"Depthwise Layer: {depthwise_layer.name}")
# weights = depthwise_layer.get_weights()[0]
# print(weights)  # shape: (channels, 1, 1, depth_multiplier)

# channels = ['Ch1', 'Ch2', 'Ch3', 'Ch4']

# depth_multiplier = weights.shape[-1]
# for i in range(depth_multiplier):
#     channel_weights = weights[:, 0, 0, i]
#     plt.bar(channels, channel_weights)
#     plt.title(f'Spatial (Depthwise) Filter Weight Analysis')
#     plt.ylabel('Weight')
#     plt.ylim(-0.4, 0.4)
#     #if i < 1: 
#         #plt.savefig("plots/eegnet_fft_model_1_regular_eegarch_12500samples_depthwiseweights.png", dpi=300, bbox_inches='tight')
#     plt.show()

# #################################Fine Tuning with Channel Weights#################################
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt
# y_combined = np.argmax(y_combined, axis=1)
# y_train = np.argmax(y_train, axis=1)
# y_new = np.argmax(y_new, axis=1)
# print(y_train.shape, X_train.shape)
# # === Assume you already have X_test and y_test prepared
# # X_test shape: (samples, 4, 60, 1)
# # y_test shape: (samples,) — assuming sparse categorical

# # # ========== 1. CHANNEL WEIGHTED INPUT ==========
# # X_weighted = X_combined.copy()
# # X_weighted[:, 0, :, :] *= 1.5  # Boost Channel 1
# # X_weighted[:, 3, :, :] *= 1.5  # Boost Channel 3

# # # Predict
# # y_pred_weighted = np.argmax(model2.predict(X_weighted), axis=1)

# # # Metrics
# # print("\n📊 Evaluation with Channel Weighted Input")
# # print("Accuracy:", accuracy_score(y_combined, y_pred_weighted))
# # print("Classification Report:\n", classification_report(y_combined, y_pred_weighted))
# # cm_weighted = confusion_matrix(y_combined, y_pred_weighted)

# # # Plot
# # plt.figure(figsize=(6, 5))
# # sns.heatmap(cm_weighted, annot=True, fmt="d", cmap="Blues")
# # plt.title("Confusion Matrix - Channel Weighted Input")
# # plt.xlabel("Predicted")
# # plt.ylabel("True")
# # plt.tight_layout()
# # plt.show()

# # # ========== 2. NOISE-AUGMENTED INPUT (only during eval) ==========
# # X_augmented = X_new_flat.copy()
# # noise_scale = 0.05  # Adjust for sensitivity

# # # Add small Gaussian noise to important channels
# # noise_ch1 = np.random.normal(0, noise_scale, size=X_augmented[:, 1, :, :].shape)
# # noise_ch3 = np.random.normal(0, noise_scale, size=X_augmented[:, 3, :, :].shape)

# # X_augmented[:, 0, :, :] += noise_ch1
# # X_augmented[:, 3, :, :] += noise_ch3

# # # Predict
# # y_pred_augmented = np.argmax(model2.predict(X_augmented), axis=1)

# # # Metrics
# # print("\n📊 Evaluation with Augmented Input (Noise in Important Channels)")
# # print("Accuracy:", accuracy_score(y_new, y_pred_augmented))
# # print("Classification Report:\n", classification_report(y_new, y_pred_augmented))
# # cm_augmented = confusion_matrix(y_new, y_pred_augmented)

# # # Plot
# # plt.figure(figsize=(6, 5))
# # sns.heatmap(cm_augmented, annot=True, fmt="d", cmap="Greens")
# # plt.title("Confusion Matrix - Augmented Input")
# # plt.xlabel("Predicted")
# # plt.ylabel("True")
# # plt.tight_layout()
# # plt.show()



# # ###############################3. OLD DATA EVALUATION

# # X_augmented = X_train_flat.copy()
# # noise_scale = 0.05  # Adjust for sensitivity

# # # Add small Gaussian noise to important channels
# # noise_ch1 = np.random.normal(0, noise_scale, size=X_augmented[:, 1, :, :].shape)
# # noise_ch3 = np.random.normal(0, noise_scale, size=X_augmented[:, 3, :, :].shape)

# # X_augmented[:, 0, :, :] += noise_ch1
# # X_augmented[:, 3, :, :] += noise_ch3

# # # Predict
# # y_pred_augmented = np.argmax(model2.predict(X_augmented), axis=1)

# # # Metrics
# # print("\n📊 Evaluation with Augmented Input (Noise in Important Channels)")
# # print("Accuracy:", accuracy_score(y_train, y_pred_augmented))
# # print("Classification Report:\n", classification_report(y_train, y_pred_augmented))
# # cm_augmented = confusion_matrix(y_train, y_pred_augmented)

# # # Plot
# # plt.figure(figsize=(6, 5))
# # sns.heatmap(cm_augmented, annot=True, fmt="d", cmap="Greens")
# # plt.title("Confusion Matrix - Augmented Input")
# # plt.xlabel("Predicted")
# # plt.ylabel("True")
# # plt.tight_layout()
# # plt.show()

# # ###############################3. Combined EVALUATION

# # X_augmented = X_combined.copy()
# # noise_scale = 0.05  # Adjust for sensitivity

# # # Add small Gaussian noise to important channels
# # noise_ch1 = np.random.normal(0, noise_scale, size=X_augmented[:, 1, :, :].shape)
# # noise_ch3 = np.random.normal(0, noise_scale, size=X_augmented[:, 3, :, :].shape)

# # X_augmented[:, 0, :, :] += noise_ch1
# # X_augmented[:, 3, :, :] += noise_ch3

# # # Predict
# # y_pred_augmented = np.argmax(model2.predict(X_augmented), axis=1)

# # # Metrics
# # print("\n📊 Evaluation with Augmented Input (Noise in Important Channels)")
# # print("Accuracy:", accuracy_score(y_combined, y_pred_augmented))
# # print("Classification Report:\n", classification_report(y_combined, y_pred_augmented))
# # cm_augmented = confusion_matrix(y_combined, y_pred_augmented)

# # # Plot
# # plt.figure(figsize=(6, 5))
# # sns.heatmap(cm_augmented, annot=True, fmt="d", cmap="Greens")
# # plt.title("Confusion Matrix - Augmented Input")
# # plt.xlabel("Predicted")
# # plt.ylabel("True")
# # plt.tight_layout()
# # plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, confusion_matrix

# # === 1. New Data Evaluation ===
# X_aug_new = X_new_flat.copy()
# noise_scale = 0.05

# X_aug_new[:, 0, :, :] += np.random.normal(0, noise_scale, size=X_aug_new[:, 1, :, :].shape)
# X_aug_new[:, 3, :, :] += np.random.normal(0, noise_scale, size=X_aug_new[:, 3, :, :].shape)

# y_pred_new = np.argmax(model2.predict(X_aug_new), axis=1)
# acc_new = accuracy_score(y_new, y_pred_new)
# cm_new = confusion_matrix(y_new, y_pred_new)

# # === 2. Old Data Evaluation ===
# X_aug_old = X_train_flat.copy()
# X_aug_old[:, 0, :, :] += np.random.normal(0, noise_scale, size=X_aug_old[:, 1, :, :].shape)
# X_aug_old[:, 3, :, :] += np.random.normal(0, noise_scale, size=X_aug_old[:, 3, :, :].shape)

# y_pred_old = np.argmax(model2.predict(X_aug_old), axis=1)
# acc_old = accuracy_score(y_train, y_pred_old)
# cm_old = confusion_matrix(y_train, y_pred_old)

# # === 3. Combined Data Evaluation ===
# X_aug_combined = X_combined.copy()
# X_aug_combined[:, 0, :, :] += np.random.normal(0, noise_scale, size=X_aug_combined[:, 1, :, :].shape)
# X_aug_combined[:, 3, :, :] += np.random.normal(0, noise_scale, size=X_aug_combined[:, 3, :, :].shape)

# y_pred_combined = np.argmax(model2.predict(X_aug_combined), axis=1)
# acc_combined = accuracy_score(y_combined, y_pred_combined)
# cm_combined = confusion_matrix(y_combined, y_pred_combined)

# # === Plot side-by-side with accuracies in titles ===
# fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# sns.heatmap(cm_new, annot=True, fmt="d", cmap="Greens", ax=axs[0])
# axs[0].set_title(f"New Data - Augmented\nAccuracy: {acc_new:.2%}")
# axs[0].set_xlabel("Predicted")
# axs[0].set_ylabel("True")

# sns.heatmap(cm_old, annot=True, fmt="d", cmap="Greens", ax=axs[1])
# axs[1].set_title(f"Old Data - Augmented\nAccuracy: {acc_old:.2%}")
# axs[1].set_xlabel("Predicted")
# axs[1].set_ylabel("True")

# sns.heatmap(cm_combined, annot=True, fmt="d", cmap="Greens", ax=axs[2])
# axs[2].set_title(f"Combined Data - Augmented\nAccuracy: {acc_combined:.2%}")
# axs[2].set_xlabel("Predicted")
# axs[2].set_ylabel("True")

# plt.tight_layout()
# plt.savefig("plots/confusion_matrices_with_accuracies_combineddata.png")
# plt.show()
