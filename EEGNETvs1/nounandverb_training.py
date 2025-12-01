# train_eegnet.py
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense, MaxPooling2D)
from sklearn.model_selection import train_test_split

# === EEGNet Model ===
def EEGNet(nb_classes, Chans=4, Samples=60, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
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
with open("eeg_fft_nounvsverb_JASON_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["data"])    # shape: (50, 250, 4, 60) 
y = np.array(data["labels"])    # shape: (12500,)

X = X.reshape(-1, 4, 60) # --> (50, 250, 4, 60) --> (12500, 4, 60)

# === Normalize (per trial)
X = (X - X.mean(axis=2, keepdims=True)) / X.std(axis=2, keepdims=True)

# === Reshape for CNN input
X = X[..., np.newaxis]     # shape: (12500, 4, 60, 1)

# === Apply static channel reweighting to reduce artifact influence
# TP channels (1 & 2): weight = 1.0
# Frontal channels (3 & 4): weight = 0.6
channel_weights = np.array([1, 0.6, 0.6, 1]).reshape(1, 4, 1, 1)
X = X #* channel_weights  # Apply to all samples but dont use for testing

print("Unique labels in y:", np.unique(y))
y = y - 1
# === One-hot encode labels
y = to_categorical(y, num_classes=2)

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# === Build and Train EEGNet
# model = EEGNet(nb_classes=2, Chans=4, Samples=60)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_test, y_test), verbose=1)

# === Save model (optional)
import time
# model.save(f"eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples{time.strftime('%Y%m%d-%H%M')}.h5")
# print("\n✅ Model trained and saved")
model = load_model("eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5")  # Load your trained model here
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")
print(model.summary())


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
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt

# # # # === Get predictions
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.argmax(y_test, axis=1)

# # === Accuracy
# acc = accuracy_score(y_true, y_pred)
# print(f"\n🎯 Test Accuracy: {acc:.4f}")

# # === Classification Report
# print("\n📋 Classification Report:")
# print(classification_report(y_true, y_pred))

# # === Confusion Matrix
# cm = confusion_matrix(y_true, y_pred)
# print("\n🔀 Confusion Matrix:")
# print(cm)

# # === Plot Confusion Matrix
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()

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


# ############################LIME Visualization#############################
# #=== Flatten input for LIME: (N, 4, 1000, 1) -> (N, 4000)
# from lime import lime_tabular

# X_train_flat = X_train.reshape(X_train.shape[0], -1)
# X_test_flat = X_test.reshape(X_test.shape[0], -1)

# # === Define LIME prediction wrapper
# def predict_fn(input_2d):
#     # Reshape back to (N, 4, 1000, 1)
#     reshaped = input_2d.reshape(-1, 4, 60, 1)
#     return model.predict(reshaped)

# # === Define feature names (optional but helpful for interpretability)
# feature_names = [f"ch{ch}_t{t}" for ch in range(4) for t in range(60)]

# # === Initialize LIME Tabular Explainer
# explainer = lime_tabular.LimeTabularExplainer(
#     training_data=X_train_flat,
#     feature_names=feature_names,
#     class_names=["Class 0", "Class 1", "Class 2"],
#     discretize_continuous=True,
#     mode='classification'
# )

# # === Pick one test sample to explain
# sample_index = 0
# sample = X_test_flat[sample_index]

# # === Run explanation
# explanation = explainer.explain_instance(
#     data_row=sample,
#     predict_fn=predict_fn,
#     num_features=50  # Top 10 contributing features
# )

# print(f"\n🔍 Explanation for Sample {explanation}:")

# # === Optionally, print top features to console
# print("\n🔍 Top contributing features:")
# for feature, weight in explanation.as_list():
#     print(f"{feature}: {weight:.4f}")

import matplotlib.pyplot as plt

# Your LIME explanation as a list of tuples (feature_name, weight)
lime_features = [
    ("ch0_t11 <= 0.00", -0.0830),
    ("ch0_t7 <= 0.30", 0.0807),
    ("ch3_t8 <= 0.47", -0.0764),
    ("ch2_t19 <= -0.20", -0.0606),
    ("0.11 < ch3_t11 <= 0.80", 0.0538),
    ("ch1_t9 <= 0.55", -0.0536),
    ("ch0_t19 <= -0.20", 0.0533),
    ("ch3_t25 > 4.44", 0.0520),
    ("ch0_t39 > -0.28", 0.0518),
    ("ch2_t8 <= 0.40", 0.0510),
    ("ch2_t13 <= 0.02", -0.0509),
    ("ch3_t15 <= -0.04", -0.0489),
    ("ch2_t11 <= 0.14", 0.0488),
    ("ch2_t10 <= 0.20", -0.0487),
    ("ch1_t14 <= 0.03", -0.0486),
    ("ch2_t9 <= 0.16", 0.0449),
    ("ch1_t11 <= 0.27", -0.0447),
    ("0.15 < ch0_t6 <= 0.79", -0.0444),
    ("ch2_t35 > -0.26", 0.0444),
    ("ch0_t15 <= -0.06", 0.0420),
    ("ch1_t10 <= 0.39", 0.0402),
    ("ch0_t48 > -0.34", -0.0383),
    ("-0.47 < ch0_t43 <= -0.31", -0.0378),
    ("ch3_t4 > -0.09", -0.0365),
    ("ch2_t43 > -0.34", 0.0359),
    ("ch1_t12 <= 0.16", -0.0359),
    ("ch1_t25 > 4.33", -0.0353),
    ("ch1_t35 > -0.28", -0.0344),
    ("ch0_t54 > -0.38", 0.0342),
    ("ch2_t3 > -0.39", -0.0341),
    ("ch1_t26 > 5.35", -0.0329),
    ("ch1_t56 > -0.44", -0.0314),
    ("ch1_t23 <= -0.19", -0.0311),
    ("3.27 < ch0_t25 <= 4.65", 0.0310),
    ("ch1_t8 <= 0.63", 0.0295),
    ("0.36 < ch2_t7 <= 1.08", -0.0294),
    ("-0.61 < ch0_t57 <= -0.41", -0.0293),
    ("ch2_t39 > -0.29", -0.0289),
    ("ch1_t4 > -0.05", 0.0280),
    ("ch0_t59 > -0.41", 0.0272),
    ("ch1_t36 > -0.30", -0.0271),
    ("ch1_t13 <= 0.02", 0.0270),
    ("ch1_t55 > -0.46", -0.0264),
    ("0.43 < ch2_t27 <= 0.84", 0.0264),
    ("-0.36 < ch0_t35 <= -0.25", -0.0259),
    ("ch0_t23 <= -0.22", -0.0253),
    ("0.39 < ch1_t27 <= 0.86", 0.0248),
    ("ch0_t58 > -0.41", 0.0246),
    ("0.17 < ch0_t9 <= 0.97", 0.0224),
    ("ch0_t8 <= 0.31", -0.0211)
]

# Separate names and values
features, weights = zip(*lime_features)

# Plot
# Plot vertically
plt.figure(figsize=(14, 6))
colors = ['green' if w > 0 else 'red' for w in weights]
plt.bar(features, weights, color=colors)
plt.ylabel('Contribution to Prediction')
plt.xlabel('Feature')
plt.title('Top LIME Feature Contributions (Vertical)')
plt.xticks(rotation=45, ha = "right", fontsize=6)
plt.axhline(0, color='gray', linewidth=0.8)
plt.tight_layout()
#plt.savefig("plots/eegnet_noun_verb_averagepooling_50epochs_regular_arch__12500samples_featureimportance.png")
plt.show()


# ##################GET DEPTHWISE CONVOLUTION LAYERS/WEIGHTS##################
# import matplotlib.pyplot as plt
# # Identify the depthwise layer (usually index 3)
depthwise_layer = model.get_layer("depthwise_conv2d")  # Adjust the name if necessary
print(f"Depthwise Layer: {depthwise_layer.name}")
weights = depthwise_layer.get_weights()[0]
print(weights)  # shape: (channels, 1, 1, depth_multiplier)

channels = ['T7', 'T8', 'F7', 'F8']

depth_multiplier = weights.shape[-1]
for i in range(depth_multiplier):
    channel_weights = weights[:, 0, 0, i]
    plt.bar(channels, channel_weights)
    plt.title(f'Depthwise Conv 2D Weight Analysis{i+1}')
    plt.ylabel('Weight')
    plt.ylim(-0.1, 0.1 )
    # if i > 0: 
    #     plt.savefig("plots/eegnet_noun_verb_averagepooling_50epochs_regular_arch__channelweighted[1661]_depthwiseweights.png")
    plt.show()


import re
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# # === Step 1: Raw Input (Feature : Importance)
raw_input = """
ch0_t11 <= 0.00: -0.0830
ch0_t7 <= 0.30: 0.0807
ch3_t8 <= 0.47: -0.0764
ch2_t19 <= -0.20: -0.0606
0.11 < ch3_t11 <= 0.80: 0.0538
ch1_t9 <= 0.55: -0.0536
ch0_t19 <= -0.20: 0.0533
ch3_t25 > 4.44: 0.0520
ch0_t39 > -0.28: 0.0518
ch2_t8 <= 0.40: 0.0510
ch2_t13 <= 0.02: -0.0509
ch3_t15 <= -0.04: -0.0489
ch2_t11 <= 0.14: 0.0488
ch2_t10 <= 0.20: -0.0487
ch1_t14 <= 0.03: -0.0486
ch2_t9 <= 0.16: 0.0449
ch1_t11 <= 0.27: -0.0447
0.15 < ch0_t6 <= 0.79: -0.0444
ch2_t35 > -0.26: 0.0444
ch0_t15 <= -0.06: 0.0420
ch1_t10 <= 0.39: 0.0402
ch0_t48 > -0.34: -0.0383
-0.47 < ch0_t43 <= -0.31: -0.0378
ch3_t4 > -0.09: -0.0365
ch2_t43 > -0.34: 0.0359
ch1_t12 <= 0.16: -0.0359
ch1_t25 > 4.33: -0.0353
ch1_t35 > -0.28: -0.0344
ch0_t54 > -0.38: 0.0342
ch2_t3 > -0.39: -0.0341
ch1_t26 > 5.35: -0.0329
ch1_t56 > -0.44: -0.0314
ch1_t23 <= -0.19: -0.0311
3.27 < ch0_t25 <= 4.65: 0.0310
ch1_t8 <= 0.63: 0.0295
0.36 < ch2_t7 <= 1.08: -0.0294
-0.61 < ch0_t57 <= -0.41: -0.0293
ch2_t39 > -0.29: -0.0289
ch1_t4 > -0.05: 0.0280
ch0_t59 > -0.41: 0.0272
ch1_t36 > -0.30: -0.0271
ch1_t13 <= 0.02: 0.0270
ch1_t55 > -0.46: -0.0264
0.43 < ch2_t27 <= 0.84: 0.0264
-0.36 < ch0_t35 <= -0.25: -0.0259
ch0_t23 <= -0.22: -0.0253
0.39 < ch1_t27 <= 0.86: 0.0248
ch0_t58 > -0.41: 0.0246
0.17 < ch0_t9 <= 0.97: 0.0224
ch0_t8 <= 0.31: -0.0211
"""

##=== Step 2: Extract feature -> importance
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
plt.figure(figsize=(7, 4))
channels = sorted(channel_avg.keys(), key=lambda x: int(x[2:]))
avg_vals = [channel_avg[ch] for ch in channels]

plt.bar(channels, avg_vals, color='skyblue')
plt.title("Average LIME Contribution per Channel For Model Unweighted EEGNet Model")
plt.xlabel("Channel")
plt.ylabel("Average Contribution")
plt.axhline(0, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig("plots/eegnet_noun_verb_averagepooling_50epochs_regular_arch__12500samples_avgchannelimportance.png")
plt.show()