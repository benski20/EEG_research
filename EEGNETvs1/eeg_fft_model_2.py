from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                        BatchNormalization, Activation, AveragePooling2D,
                                        Dropout, Flatten, Dense, TimeDistributed, LSTM)
from tensorflow.keras.utils import to_categorical
import pickle
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense, TimeDistributed, LSTM)

def EEGNet_time_distributed(nb_classes, time_steps=250, Chans=4, Samples=60, dropoutRate=0.5):
    input_main = Input(shape=(time_steps, Chans, Samples, 1))  # (250, 4, 60, 1)
    
    # Apply EEGNet CNN on each time step independently
    x = TimeDistributed(Conv2D(8, (1, 64), padding='same', use_bias=False))(input_main)
    x = TimeDistributed(BatchNormalization())(x)
    
    x = TimeDistributed(DepthwiseConv2D((Chans, 1), use_bias=False, depth_multiplier=2))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('elu'))(x)
    x = TimeDistributed(AveragePooling2D((1,4)))(x)
    x = TimeDistributed(Dropout(dropoutRate))(x)
    
    x = TimeDistributed(SeparableConv2D(16, (1, 16), padding='same', use_bias=False))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(Activation('elu'))(x)
    x = TimeDistributed(AveragePooling2D((1,8)))(x)
    x = TimeDistributed(Dropout(dropoutRate))(x)
    
    x = TimeDistributed(Flatten())(x)  # shape (batch, time_steps, features)
    
    # Now use LSTM over time steps to capture temporal dependencies
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(dropoutRate)(x)
    
    output = Dense(nb_classes, activation='softmax')(x)
    
    return Model(inputs=input_main, outputs=output)

# === Load Data ===
with open("eeg_fft_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["data"])
#X = np.mean(X, axis=1)    # shape: (N, 4, 60) --> taking the mean over 250 samples per trial
#X = X.reshape(-1, 4, 60) 
print(X.shape) # Reshape to (N, 4, 60) if needed
y = np.array(data["labels"])    # shape: (N,)
# y shape: (12500,)
y = y.reshape(50, 250)  # (trials, snapshots_per_trial) --> ONLY DO WHEN TAKING MEAN OVER SAMPLES

# Take the first label per trial (if all are the same per trial)
y = y[:, 0] #--> ONLY DO WHEN TAKING MEAN OVER SAMPLES

# === Normalize (per trial)
X = (X - X.mean(axis=2, keepdims=True)) / X.std(axis=2, keepdims=True)

# === Reshape for CNN input
X = X[..., np.newaxis]     # shape: (N, 4, 60, 1)

# === Apply static channel reweighting to reduce artifact influence
# TP channels (1 & 2): weight = 1.0
# Frontal channels (3 & 4): weight = 0.6
##channel_weights = np.array([0.5, 0.5, 0, 1]).reshape(1, 4, 1, 1)
X = X ##* channel_weights  # Apply to all samples but dont use for testing


# === One-hot encode labels
y = to_categorical(y, num_classes=3)

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# === Build and Train EEGNet
# model = EEGNet_time_distributed(nb_classes=3, time_steps=250, Chans=4, Samples=60)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_test, y_test), verbose=1)

# === Save model (optional)
import time
# model.save(f"eegnet_fft_model_2_lstm_fft_regularshape_{time.strftime('%Y%m%d-%H%M')}.h5")
# print("\n✅ Model trained and saved")
model = load_model("eegnet_fft_model_2_lstm_fft_regularshape_20250802-0250.h5")  # Load your trained model here
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



# #######confusion matrix and classification report for TRAINING DATA#########
# Get model predictions on training data
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


####################COMPUTE AVERAGE METRICS AND CONFUSION MATRIX################
# import os
# from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score
# from sklearn.model_selection import train_test_split

# y_train_labels = np.argmax(y_train, axis=1)
# y_test_labels = np.argmax(y_test, axis=1)

# # Model file list
# model_files = [
#     #"eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5",
#     #"eegnet_fft_model_2_lstm_fft_regularshape_20250802-0250.h5"
#     #"eegnet_fft_model_1_regular_eegarch_mean250samples_20250802-0120.h5"
#     "eegnet_fft_model_3_attention_model20250802-1145.h5",
    
# ]

# output_dir = "fftmodel_random_sample_evaluation_results"
# os.makedirs(output_dir, exist_ok=True)

# # === Extract weights from filename like [5501] → [0.5, 0.5, 0, 1.0] ===
# import re
# import numpy as np

# def extract_weights_from_filename(filename):
#     import re
#     match = re.search(r"\[(\d{4})\]", filename)
#     if not match:
#         return np.ones((1, 4, 1, 1))  # fallback: all weights = 1.0

#     digits = match.group(1)
#     weights = [1.0 if d == '1' else int(d)/10.0 for d in digits]
#     return np.array(weights).reshape(1, 4, 1, 1)


# # === Evaluate one model with its correct channel weighting ===
# def evaluate_model(model_path, n_runs=100):
#     print(f"Evaluating {model_path} for {n_runs} runs...")
    
#     # if model_path == "eegnet_fft_model_1_regular_eegarch_mean250samples_20250802-0120.h5":
#     #     X_train_w = np.mean(X_train, axis=1)
#     #     X_test_w = np.mean(X_test, axis=1)
#     #     y_train_labels = y.reshape(50, 250) 
#     #     y_train_labels = y_train_labels[:, 0]
#     #     y_test_labels = y.reshape(50, 250)
#     #     y_test_labels = y_test_labels[:, 0]
#     # if model_path == "eegnet_fft_model_2_lstm_fft_regularshape_20250802-0250.h5":
#     #     X_train_w = X_train
#     #     X_test_w = X_test
#     #     y_train_labels = y.reshape(50, 250) 
#     #     y_train_labels = y_train_labels[:, 0]
#     #     y_test_labels = y.reshape(50, 250)
#     #     y_test_labels = y_test_labels[:, 0]
#     # if model_path == "eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5":
#     #     X_train_w = X_train.reshape(-1, 4, 60)
#     #     X_test_w = X_test.reshape(-1, 4, 60)
#     #     y_train_labels = np.argmax(y_train, axis=1)
#     #     y_test_labels = np.argmax(y_test, axis=1)
    
#     model = load_model(model_path, custom_objects={"CannelAttentionModel": ChannelAttentionModel})

#     X_train_w = X_train 
#     X_test_w = X_test

#     train_accs, test_accs = [], []
#     f1s, recalls = [], []
#     cms = []

#     for _ in range(n_runs):
#         idx = np.random.choice(len(X_test_w), int(0.8 * len(X_test_w)), replace=True)
#         X_test_sample = X_test_w[idx]
#         y_test_sample = y_test_labels[idx]
#         print(idx)

#         train_pred_probs = model.predict(X_train_w, verbose=0)
#         train_preds = np.argmax(train_pred_probs, axis=1)
#         train_acc = accuracy_score(y_train_labels, train_preds)
#         train_accs.append(train_acc)

#         test_pred_probs = model.predict(X_test_sample, verbose=0)
#         test_preds = np.argmax(test_pred_probs, axis=1)

#         test_acc = accuracy_score(y_test_sample, test_preds)
#         recall = recall_score(y_test_sample, test_preds, average='macro')
#         f1 = f1_score(y_test_sample, test_preds, average='macro')
#         cm = confusion_matrix(y_test_sample, test_preds, labels=[0, 1, 2])

#         test_accs.append(test_acc)
#         recalls.append(recall)
#         f1s.append(f1)
#         cms.append(cm)
#         print(test_accs)

#     avg_train_acc = np.mean(train_accs)
#     avg_test_acc = np.mean(test_accs)
#     avg_recall = np.mean(recalls)
#     avg_f1 = np.mean(f1s)
#     avg_cm = np.mean(cms, axis=0)

#     return {
#         "train_accuracy": avg_train_acc,
#         "test_accuracy": avg_test_acc,
#         "recall": avg_recall,
#         "f1_score": avg_f1,
#         "avg_confusion_matrix": avg_cm
#     }

# # === Save metrics and confusion matrix ===
# def save_results(results, model_name):
#     base_name = os.path.splitext(os.path.basename(model_name))[0]
#     metrics_file = os.path.join(output_dir, f"{base_name}_metrics.txt")
#     cm_file = os.path.join(output_dir, f"{base_name}_confusion_matrix.npy")

#     with open(metrics_file, "w") as f:
#         for k, v in results.items():
#             if k != "avg_confusion_matrix":
#                 f.write(f"{k}: {v:.4f}\n")

#     np.save(cm_file, results["avg_confusion_matrix"])

#     print(f"Saved metrics to {metrics_file}")
#     print(f"Saved confusion matrix to {cm_file}")

# # === Run for all models ===
# for mf in model_files:
#     results = evaluate_model(mf)
#     save_results(results, mf)