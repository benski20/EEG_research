import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense, MaxPooling2D)
from sklearn.model_selection import train_test_split

# --- Channel Attention Layer ---
class ChannelAttention(layers.Layer):
    def __init__(self, channels, reduction_ratio=2, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.channels = channels                   # store attributes
        self.reduction_ratio = reduction_ratio

        self.global_pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(channels // reduction_ratio, activation='relu')
        self.fc2 = layers.Dense(channels, activation='sigmoid')

    def call(self, inputs):
        # inputs: (B, C, T, 1)
        avg_pool = self.global_pool(inputs)  # (B, C)
        x = self.fc1(avg_pool)
        x = self.fc2(x)
        scale = tf.reshape(x, [-1, inputs.shape[1], 1, 1])  # (B, C, 1, 1)
        return inputs * scale

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({
            "channels": self.channels,
            "reduction_ratio": self.reduction_ratio,
        })
        return config

# --- Temporal Attention Layer ---
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(TemporalAttention, self).__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')

    def call(self, x):
        # x shape: (B, C, T, F)
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)  # (B, C, T, 1)
        attn = self.conv(avg_pool)  # (B, C, T, 1)
        return x * attn

    def get_config(self):
        config = super(TemporalAttention, self).get_config()
        # No extra attributes here, so just return base config
        return config


# --- EEGNet with Attention ---
def EEGNet_with_Attention(nb_classes, Chans=4, Samples=1000,
                          dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16):
    input_main = Input(shape=(Chans, Samples, 1))

    ## Block 1
    x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input_main)
    x = BatchNormalization()(x)

    # Add Channel Attention
    #x = ChannelAttention(channels=Chans)(x)

    # Depthwise Convolution (spatial filtering across channels)
    x = DepthwiseConv2D((Chans, 1), depth_multiplier=D, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)

    # Add Temporal Attention
    x = TemporalAttention()(x)

    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropoutRate)(x)

    ## Block 2
    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)

    ## Classification
    x = Flatten(name='flatten')(x)
    output = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=input_main, outputs=output)

# === Load Data ===
with open("eeg_raw_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["X"])    # shape: (N, 4, 1000)
y = np.array(data["y"])    # shape: (N,)

# === Normalize (per trial)
X = (X - X.mean(axis=2, keepdims=True)) / X.std(axis=2, keepdims=True)

# === Reshape for CNN input
X = X[..., np.newaxis]     # shape: (N, 4, 1000, 1)

# === Apply static channel reweighting to reduce artifact influence
# TP channels (1 & 2): weight = 1.0
# Frontal channels (3 & 4): weight = 0.6
#channel_weights = np.array([0.5, 0.5, 0, 1]).reshape(1, 4, 1, 1) ##not needed for attention model
#X = X ##* channel_weights  # Apply to all samples but dont use for testing

# === One-hot encode labels
y = to_categorical(y, num_classes=3)

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# === Build and Train EEGNet
# model = EEGNet_with_Attention(nb_classes=3, Chans=4, Samples=1000)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_test, y_test), verbose=1)

import time
# model.save(f"eegnet_model_1_with_just_temporal_attention_{time.strftime('%Y%m%d-%H%M')}.h5")
# print("\n✅ Model trained and saved")
#model = load_model("eegnet_model_1_with_channel_and_temporal_attention_2318.h5")  # Load your trained model here
model = load_model( ##load trained model with custom artifact reduction layers
    "eegnet_model_1_with_just_channel_attention_20250801-2158.h5",
    custom_objects={
        "ChannelAttention": ChannelAttention,
        ##"TemporalAttention": TemporalAttention,
    }
)
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"\n✅ Test Accuracy: {accuracy:.4f}")
# print(model.summary())

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

########################COMPUTE AVERAGE CONFUSION MATRIX FOR ALL TRIALS & ALL METRICS########################
# from tensorflow.keras.models import load_model
# from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
# import numpy as np
# import os

# def evaluate_attention_model(model_path, X_train, y_train, X_test, y_test, n_runs=100):
#     print(f"Evaluating {model_path} for {n_runs} runs...")

#     if model_path == "eegnet_model_1_with_channel_and_temporal_attention_2318.h5": 
#         model = load_model(model_path, custom_objects={
#             'ChannelAttention': ChannelAttention,
#             'TemporalAttention': TemporalAttention
#         })
#     if model_path == "eegnet_model_1_with_just_temporal_attention_20250801-2200.h5":
#         model = load_model(model_path, custom_objects={
#             'TemporalAttention': TemporalAttention
#         })
#     if model_path == "eegnet_model_1_with_just_channel_attention_20250801-2158.h5":
#         model = load_model(model_path, custom_objects={
#             'ChannelAttention': ChannelAttention
#         })
    
#     X_train_w = X_train
#     X_test_w = X_test

#     train_accs, test_accs = [], []
#     f1s, recalls = [], []
#     cms = []

#     for _ in range(n_runs):
#         # Predict on train and test sets
#         idx = np.random.choice(len(X_test_w), int(0.8 * len(X_test_w)), replace=True)
#         X_test_sample = X_test[idx]
#         y_test_sample = y_test[idx]
#         print(idx)

#         train_pred_probs = model.predict(X_train, verbose=0)  # Train set not sampled
#         train_preds = np.argmax(train_pred_probs, axis=1)
#         train_acc = accuracy_score(np.argmax(y_train, axis=1), train_preds)
#         train_accs.append(train_acc)

#         test_pred_probs = model.predict(X_test_sample, verbose=0)
#         test_preds = np.argmax(test_pred_probs, axis=1)

#         test_acc = accuracy_score(np.argmax(y_test_sample, axis=1), test_preds)
#         recall = recall_score(np.argmax(y_test_sample, axis=1), test_preds, average='macro')
#         f1 = f1_score(np.argmax(y_test_sample, axis=1), test_preds, average='macro')
#         cm = confusion_matrix(np.argmax(y_test_sample, axis=1), test_preds, labels=[0, 1, 2])

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

# # === Run evaluation for all attention models ===
# attention_model_files = [
#     "eegnet_model_1_with_channel_and_temporal_attention_2318.h5",
#     "eegnet_model_1_with_just_temporal_attention_20250801-2200.h5",
#     "eegnet_model_1_with_just_channel_attention_20250801-2158.h5"
# ]

# output_dir = "model_attention_evaluation_results"
# os.makedirs(output_dir, exist_ok=True)

# for model_file in attention_model_files:
#     results = evaluate_attention_model(model_file, X_train, y_train, X_test, y_test, n_runs=100)
#     save_results(results, model_file)



################################## # code to extract channel attention weights from a single sample from trained model#################
import numpy as np
from scipy.special import expit, softmax  # sigmoid function

# Assume you have loaded your model and have X_test available

# Get the channel attention layer weights
channel_attention_layer = model.get_layer(name='channel_attention')
weights = channel_attention_layer.get_weights()

# Extract weights and biases
W_fc1 = weights[0]  # shape (8, 2)
b_fc1 = weights[1]  # shape (2,)
W_fc2 = weights[2]  # shape (2, 4)
b_fc2 = weights[3]  # shape (4,)
from tensorflow.keras.models import Model


# Get the layer right before channel_attention
prev_layer = model.layers[model.layers.index(channel_attention_layer) - 1]

intermediate_model = Model(inputs=model.input, outputs=prev_layer.output)

# Get intermediate output for a sample input with raw shape (1, 4, time, 1)
sample_input = X_test[np.random.randint(len(X_test))]
sample_input = np.expand_dims(sample_input, axis=0)  # add batch dim

intermediate_output = intermediate_model.predict(sample_input)  # This should have shape (1, 8, T, 1) or so

print("Intermediate output shape feeding into channel attention:", intermediate_output.shape)

intermediate_output_perm = np.transpose(intermediate_output, (0, 3, 2, 1))  # shape: (1, 8, 1000, 4)
pooled_input = np.mean(intermediate_output_perm, axis=(2, 3))  # pool time and last dim -> (1, 8)

fc1_out = np.dot(pooled_input, W_fc1) + b_fc1  # (1, 2)
fc1_out_relu = np.maximum(fc1_out, 0)

logits = np.dot(fc1_out_relu, W_fc2) + b_fc2  # (1, 4)
attention_scores = expit(logits)
attention_scores = softmax(attention_scores, axis=-1)  # Normalize to sum to 1
print("Channel Attention Scores:", attention_scores.flatten())


##################CODE TO COMPUTE AVERAGE ATTENTION SCORES FOR THE TRAINING DATASET##################
# import numpy as np
# from scipy.special import expit, softmax
# from tensorflow.keras.models import Model

# # Get channel attention weights from the model
# channel_attention_layer = model.get_layer(name='channel_attention')
# weights = channel_attention_layer.get_weights()

# W_fc1 = weights[0]  # shape (8, 2)
# b_fc1 = weights[1]  # shape (2,)
# W_fc2 = weights[2]  # shape (2, 4)
# b_fc2 = weights[3]  # shape (4,)

# # Get the layer right before channel_attention layer to extract intermediate output
# prev_layer = model.layers[model.layers.index(channel_attention_layer) - 1]
# intermediate_model = Model(inputs=model.input, outputs=prev_layer.output)

# attention_scores_all = []

# for i in range(len(X_test)):
#     sample_input = X_test[i]
#     sample_input = np.expand_dims(sample_input, axis=0)  # add batch dimension

#     # Get intermediate output feeding into channel attention
#     intermediate_output = intermediate_model.predict(sample_input, verbose=0)  # shape (1, C, T, 1)
    
#     # Permute to (1, 8, T, 4) — adjust if necessary to match your data shape
#     intermediate_output_perm = np.transpose(intermediate_output, (0, 3, 2, 1))
    
#     # Global average pooling over time and last dim: (1, 8)
#     pooled_input = np.mean(intermediate_output_perm, axis=(2, 3))
    
#     # Pass through fc1 with ReLU
#     fc1_out = np.dot(pooled_input, W_fc1) + b_fc1  # shape (1, 2)
#     fc1_out_relu = np.maximum(fc1_out, 0)
    
#     # Pass through fc2 and sigmoid activation
#     logits = np.dot(fc1_out_relu, W_fc2) + b_fc2  # shape (1, 4)
#     attention_scores = expit(logits)
    
#     # Normalize to sum to 1 for interpretability
#     attention_scores = softmax(attention_scores, axis=-1)  # shape (1, 4)
    
#     attention_scores_all.append(attention_scores.flatten())

# # Compute average attention scores over all training samples
# avg_attention_scores = np.mean(attention_scores_all, axis=0)

# print("Average Channel Attention Scores over Training Set:", avg_attention_scores)


###################################################LAYER WEIGHTS FOR SEPARABLE CONVOLUTION#####################################
# # Identify the depthwise layer (usually index 3)
depthwise_layer = model.get_layer("depthwise_conv2d")  # Adjust name if necessary
print(f"Depthwise Layer: {depthwise_layer.name}")
weights = depthwise_layer.get_weights()[0]
print(weights)  # shape: (channels, 1, 1, depth_multiplier)

channels = ['Temporal-Parietal (TP-7/TP-8)', 'Temporal-Parietal', 'Frontal (F7/F8)', 'Frontal(F7/F8)']

depth_multiplier = weights.shape[-1]
for i in range(depth_multiplier):
    channel_weights = weights[:, 0, 0, i]
    plt.bar(channels, channel_weights)
    plt.title(f'Separable Conv 2D {i+1}')
    plt.ylabel('Weight')
    plt.ylim(-0.1, 0.1)
    plt.show()