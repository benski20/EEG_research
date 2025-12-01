from tensorflow.keras.models import Model, load_model
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense, TimeDistributed, LSTM)

class ChannelAttentionModel(tf.keras.Model):
    def __init__(self, num_channels=4, fft_bins=60, num_classes=3, attention_dim=32):
        super(ChannelAttentionModel, self).__init__()

        self.conv1 = tf.keras.layers.Conv1D(16, kernel_size=3, padding='same', activation='relu')
        self.pool = tf.keras.layers.GlobalAveragePooling1D()

        self.attention_dense1 = tf.keras.layers.Dense(attention_dim, activation='relu')
        self.attention_dense2 = tf.keras.layers.Dense(1, activation=None)  # softmax applied later

        self.classifier_dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.classifier_dense2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        num_channels = inputs.shape[1]

        x = tf.reshape(inputs, (-1, inputs.shape[2], 1))
        x = self.conv1(x)
        x = self.pool(x)
        x = tf.reshape(x, (batch_size, num_channels, -1))

        attn = self.attention_dense1(x)
        attn = self.attention_dense2(attn)
        attn = tf.squeeze(attn, axis=-1)
        attn_weights = tf.nn.softmax(attn, axis=1)

        weighted_sum = tf.reduce_sum(x * tf.expand_dims(attn_weights, -1), axis=1)

        out = self.classifier_dense1(weighted_sum)
        out = self.classifier_dense2(out)

        return out  # <- only return predictions

    def get_attention(self, inputs):
        # Separate method to extract attention weights
        batch_size = tf.shape(inputs)[0]
        num_channels = inputs.shape[1]

        x = tf.reshape(inputs, (-1, inputs.shape[2], 1))
        x = self.conv1(x)
        x = self.pool(x)
        x = tf.reshape(x, (batch_size, num_channels, -1))

        attn = self.attention_dense1(x)
        attn = self.attention_dense2(attn)
        attn = tf.squeeze(attn, axis=-1)
        attn_weights = tf.nn.softmax(attn, axis=1)
        return attn_weights



import pickle
# === Load your data (replace with actual loading code) ===
with open("eeg_fft_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["data"])   # Expected shape: (N, 4, 60)
y = np.array(data["labels"]) # Integer labels

X = X.reshape(-1, 4, 60)

# Normalize each sample across fft bins (per channel)
X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)

# One-hot encode labels
num_classes = 3
y_cat = to_categorical(y, num_classes=num_classes)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y)

# Create the model instance
# model = ChannelAttentionModel(num_channels=4, fft_bins=60, num_classes=num_classes)

# # Build model (required to see summary)
# # Compile
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# # Train
# model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_test, y_test), verbose=1)

# # Evaluate
# loss, acc = model.evaluate(X_test, y_test)
# print(f"Test accuracy: {acc:.4f}")

# Get attention weights on test data (for interpretability)
# Get class predictions
# predictions = model(X_test, training=False)

# # Get attention weights separately
# attention_weights = model.get_attention(X_test)
# avg_attention = tf.reduce_mean(attention_weights, axis=0).numpy()
# print("Average attention weights per channel:", avg_attention)


# === Build and Train EEGNet
# model = EEGNet_time_distributed(nb_classes=3, time_steps=250, Chans=4, Samples=60)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_test, y_test), verbose=1)

# === Save model (optional)
import time
#model.save(f"eegnet_fft_model_3_attention_model{time.strftime('%Y%m%d-%H%M')}.h5")
#print("\n✅ Model trained and saved")
model = load_model("eegnet_fft_model_3_attention_model20250802-1145.h5")  # Load your trained model here
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
# #Get model predictions on training data
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