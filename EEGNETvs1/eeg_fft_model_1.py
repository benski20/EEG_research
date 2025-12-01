# train_eegnet.py
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense, TimeDistributed, LSTM)

from sklearn.model_selection import train_test_split

# === EEGNet Model ===
def EEGNet(nb_classes, Chans=4, Samples=1000, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
    input_main = Input(shape=(Chans, Samples, 1))

    x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input_main)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((Chans, 1), use_bias=False,
                        depth_multiplier=D)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropoutRate)(x)

    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)

    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=input_main, outputs=x)

# === Load Data ===
with open("eeg_raw_nounvsverb_JASON_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["X"])
#X = np.mean(X, axis=1)    # shape: (N, 4, 60) --> taking the mean over 250 samples per trial
#X = X.reshape(-1, 4, 60) 
print(X.shape) # Reshape to (N, 4, 60) if needed
y = np.array(data["y"])    # shape: (N,)
# y shape: (12500,)
y = y - 1
#y = y.reshape(50, 250)  # (trials, snapshots_per_trial) --> ONLY DO WHEN TAKING MEAN OVER SAMPLES

# Take the first label per trial (if all are the same per trial)
#y = y[:, 0] --> ONLY DO WHEN TAKING MEAN OVER SAMPLES

# === Normalize (per trial)
X = (X - X.mean(axis=2, keepdims=True)) / X.std(axis=2, keepdims=True)

# === Reshape for CNN input
X = X[..., np.newaxis]     # shape: (N, 4, 60, 1)

# === Apply static channel reweighting to reduce artifact influence
# TP channels (1 & 2): weight = 1.0
# Frontal channels (3 & 4): weight = 0.6
channel_weights = np.array([1, 6, 6, 1]).reshape(1, 4, 1, 1)
X = X * channel_weights  # Apply to all samples but dont use for testing


# === One-hot encode labels
y = to_categorical(y, num_classes=2)

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, stratify=y)

# === Build and Train EEGNet
# model = EEGNet(nb_classes=3, Chans=4, Samples=60)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_test, y_test), verbose=1)

# === Save model (optional)
import time
# model.save(f"eegnet_fft_model_1_regular_eegarch_12500samples_{time.strftime('%Y%m%d-%H%M')}.h5")
# print("\n✅ Model trained and saved")
model = load_model("eegnet_noun_verb_averagepooling_50epochs_regular_arch__channelweighted[1661]20250805-2136.h5")  # Load your trained model here
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"\n✅ Test Accuracy: {accuracy:.4f}")
# print(model.summary())


# ############-evaluate a certain trial apart of the test set-##
# import random
# # Select a random trial from the test set
# random_index = random.randint(0, len(X_test) - 1)
# print(f"Selected Trial Index: {random_index}")
# new_trial = X_test[random_index]
# new_trial_label = y_test[random_index]
# new_trial_label = np.argmax(new_trial_label)  # Convert one-hot to class index
# print(f"Trial Label: {new_trial_label}")  # Print the label for the trial


# new_trial = new_trial[np.newaxis, ..., np.newaxis]  # shape: (1, 4, 1000, 1)
#  # Print the one-hot encoded label for the trial

# prediction = model.predict(new_trial)
# predicted_class = np.argmax(prediction)

# print(f"Predicted Class: {predicted_class}")


# #######confusion matrix and classification report for TEST DATA#########
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# # # === Get predictions
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

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
#plt.savefig("plots/eegnet_fft_model_1_cm_nochanges.png", dpi=300, bbox_inches='tight')
plt.show()

# print("Evaluation", model.evaluate(X_test, y_test))
# print("Evaluation Train", model.evaluate(X_train, y_train))  # Evaluate the model on the test set



# # #######confusion matrix and classification report for TRAINING DATA#########
# # Get model predictions on training data
# y_train_pred = model.predict(X_train)
# y_train_pred_classes = np.argmax(y_train_pred, axis=1)

# # Actual labels (if one-hot encoded)
# y_train_true = np.argmax(y_train, axis=1)

# # Accuracy
# train_accuracy = accuracy_score(y_train_true, y_train_pred_classes)
# print("Train Accuracy: {:.4f}".format(train_accuracy))

# # Classification Report
# print("\n📋 Classification Report (Train):")
# print(classification_report(y_train_true, y_train_pred_classes))

# # Confusion Matrix
# print("\n📉 Confusion Matrix (Train):")
# print(confusion_matrix(y_train_true, y_train_pred_classes))

# cm = confusion_matrix(y_train_true, y_train_pred_classes)

# ## === Plot Confusion Matrix
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()

######################################LIME EXPLANATION######################################
# from lime import lime_tabular
# import tensorflow as tf

# # Flatten
# X_train_flat = X_train.reshape(X_train.shape[0], -1)
# X_test_flat = X_test.reshape(X_test.shape[0], -1)

# # Prediction wrapper
# def predict_fn(input_2d):
#     reshaped = input_2d.reshape(-1, 4, 60, 1)
#     logits = model.predict(reshaped)
#     return tf.nn.softmax(logits).numpy()  # Ensure probabilities

# # Feature names
# feature_names = [f"ch{ch}_t{t}" for ch in range(4) for t in range(60)]

# # LIME Explainer
# explainer = lime_tabular.LimeTabularExplainer(
#     training_data=X_train_flat,
#     feature_names=feature_names,
#     class_names=["Class 0", "Class 1", "Class 2"],
#     discretize_continuous=True,
#     mode='classification'
# )

# # Explain one test sample
# sample_index = 0
# sample = X_test_flat[sample_index]
# explanation = explainer.explain_instance(
#     data_row=sample,
#     predict_fn=predict_fn,
#     num_features=60
# )

# # Show explanation
# for feature, weight in explanation.as_list():
#     print(f"{feature}: {weight:.4f}")


#######################################LIME PLOT (ABBREVIATED GRAPH - 20 FEATURES#######################################
# import matplotlib.pyplot as plt

# # Feature names and importance values
# features = [
#     "ch1_t51", "ch1_t18", "ch3_t51", "ch3_t25", "ch1_t19", "ch1_t36",
#     "ch1_t40", "ch1_t6", "ch1_t47", "ch1_t52", "ch2_t9", "ch1_t33",
#     "ch3_t35", "ch1_t17", "ch0_t22", "ch3_t55", "ch1_t22", "ch0_t11",
#     "ch2_t32", "ch3_t10"
# ]

# importances = [
#      0.0547, 0.0281, 0.0251, -0.0209, 0.0200, -0.0147, -0.0138, -0.0134, -0.0130,
#      0.0123, 0.0113, -0.0109, -0.0103, 0.0103, -0.0100, -0.0099, 0.0093, -0.0092,
#     -0.0081, 0.0074
# ]

# # Sort features by absolute importance for better visualization
# sorted_indices = sorted(range(len(importances)), key=lambda i: abs(importances[i]), reverse=True)
# sorted_features = [features[i] for i in sorted_indices]
# sorted_importances = [importances[i] for i in sorted_indices]

# # Plot
# plt.figure(figsize=(12, 6))
# bars = plt.bar(range(len(sorted_importances)), sorted_importances, color='teal')
# plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
# plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
# plt.title("Top Feature Importance Scores For Model FFT-EEGNet (12500 samples)")
# plt.ylabel("Importance")
# plt.tight_layout()
# plt.grid(axis='y', linestyle='--', alpha=0.5)

# # Annotate bars
# for i, val in enumerate(sorted_importances):
#     plt.text(i, val + 0.001 * (1 if val > 0 else -1), f"{val:.3f}", ha='center', va='bottom' if val > 0 else 'top', fontsize=8)
# plt.savefig("plots/eegnet_fft_model_regular_eegarch_12500samples_feature_importance.png", dpi=300, bbox_inches='tight')
# plt.show()

# ###############################################LIME PLOT (AVERAGE CONTRIBUTION PER CHANNEL)###############################################
# import re
# from collections import defaultdict
# import numpy as np
# import matplotlib.pyplot as plt

# # === Step 1: Raw Input (Feature : Importance)
# raw_input = """
# ch1_t51 > 2.93: 0.0468
# ch1_t18 <= -0.45: -0.0199
# ch1_t39 <= -0.68: 0.0197
# ch1_t17 > 0.63: 0.0188
# ch1_t47 > 0.51: 0.0185
# ch1_t59 <= -0.77: 0.0169
# ch2_t25 <= 2.00: 0.0161
# ch3_t52 <= -0.61: 0.0136
# -0.52 < ch3_t51 <= -0.00: 0.0128
# -0.20 < ch1_t32 <= 0.48: -0.0117
# ch3_t12 > 1.50: -0.0116
# -0.59 < ch1_t27 <= -0.06: -0.0111
# ch0_t9 > 2.06: 0.0098
# ch2_t27 > 0.72: 0.0094
# ch1_t52 > 1.47: 0.0092
# ch3_t26 <= 0.94: -0.0091
# ch3_t27 <= -0.24: 0.0090
# ch3_t44 <= -0.68: -0.0090
# ch1_t37 > 0.26: -0.0088
# -0.45 < ch1_t10 <= 0.17: 0.0087
# ch0_t44 <= -0.67: -0.0085
# ch1_t46 > 0.47: -0.0083
# ch0_t15 > 0.90: 0.0082
# ch3_t17 > 0.67: -0.0082
# ch3_t47 <= -0.68: 0.0078
# ch1_t31 > 0.26: 0.0078
# ch2_t2 <= -1.11: -0.0077
# -0.81 < ch0_t55 <= -0.48: 0.0076
# 0.62 < ch2_t7 <= 2.12: -0.0076
# ch3_t1 <= -1.37: 0.0075
# ch0_t51 > 2.58: 0.0075
# -0.27 < ch1_t43 <= 0.29: -0.0071
# -0.72 < ch3_t50 <= -0.50: 0.0069
# -0.26 < ch3_t29 <= -0.17: -0.0069
# -0.69 < ch0_t33 <= -0.28: -0.0069
# -0.73 < ch1_t54 <= -0.30: -0.0068
# ch3_t39 > -0.45: 0.0068
# ch2_t46 <= -0.64: -0.0067
# -0.61 < ch2_t44 <= -0.48: 0.0067
# ch0_t24 <= -0.59: -0.0066
# ch2_t3 <= -0.82: 0.0066
# ch2_t14 > 0.87: -0.0066
# ch2_t8 > 2.34: -0.0065
# 2.10 < ch3_t9 <= 2.46: -0.0064
# ch1_t56 <= -0.72: -0.0064
# -0.31 < ch3_t31 <= -0.23: 0.0062
# -1.46 < ch1_t0 <= -1.32: -0.0061
# ch2_t58 <= -0.83: -0.0060
# 0.26 < ch0_t25 <= 0.96: 0.0060
# -0.10 < ch1_t33 <= 0.45: -0.0059
# ch0_t0 > -1.12: 0.0059
# ch1_t20 <= -0.45: 0.0058
# -1.35 < ch0_t2 <= -1.08: 0.0058
# ch0_t8 > 2.36: 0.0057
# ch1_t4 > -0.63: 0.0056
# ch1_t22 <= -0.50: -0.0055
# -0.29 < ch1_t58 <= 0.15: -0.0054
# -0.54 < ch3_t41 <= -0.47: -0.0053
# -0.27 < ch1_t12 <= 0.34: -0.0052
# -0.29 < ch3_t30 <= -0.20: -0.0045
# """

# # === Step 2: Extract feature -> importance
# pattern = re.compile(r"(ch\d+_t\d+)[^:]*:\s*(-?\d+\.\d+)")
# matches = pattern.findall(raw_input)

# features = [m[0] for m in matches]
# importances = [float(m[1]) for m in matches]

# # === Step 3: Average contribution per channel
# channel_sums = defaultdict(list)
# for feat, imp in zip(features, importances):
#     ch = feat.split("_")[0]  # ch0, ch1, etc.
#     channel_sums[ch].append(imp)

# channel_avg = {ch: np.mean(vals) for ch, vals in channel_sums.items()}

# # === Step 4: Plot
# plt.figure(figsize=(7, 4))
# channels = sorted(channel_avg.keys(), key=lambda x: int(x[2:]))
# avg_vals = [channel_avg[ch] for ch in channels]

# plt.bar(channels, avg_vals, color='skyblue')
# plt.title("Average LIME Contribution per Channel For Model FFT-EEGNet (12500 samples)")
# plt.xlabel("Channel")
# plt.ylabel("Average Contribution")
# plt.axhline(0, color='gray', linestyle='--', linewidth=1)
# plt.tight_layout()
# plt.savefig("plots/eegnet_fft_model_regular_eegarch_12500samples_averagechannel_importance.png", dpi=300, bbox_inches='tight')
# plt.show()
