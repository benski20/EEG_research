# multi_task_fusion.py - FIXED VERSION
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -------------------------
# Config
# -------------------------
EMOTION_MODEL_PATH = "eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5"
POS_MODEL_PATH = "eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5"
EMOTION_DATA_PATH = "eeg_fft_dataset.pkl"  # {"X": [...], "y": [...]}
POS_DATA_PATH = "eeg_fft_nounvsverb_JASON_dataset.pkl"          # {"X": [...], "y": [...]}

INPUT_SHAPE = (4, 60, 1)  # (channels, time, 1)
NUM_CLASSES_EMOTION = 3
NUM_CLASSES_POS = 2
FUSION_DENSE = 128
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
RANDOM_STATE = 42

# -------------------------
# Helpers
# -------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_feature_extractor(model):
    """Return a Model that maps model.input -> layer_before_dense_output.
       Strategy: try to find a layer named 'flatten' or return penultimate layer.
    """
    # try to find a Flatten layer
    flatten_layer = None
    for layer in model.layers:
        if layer.__class__.__name__.lower() == "flatten" or "flatten" in layer.name:
            flatten_layer = layer
            break

    if flatten_layer is not None:
        feature_output = flatten_layer.output
    else:
        # fallback: take layer before last Dense (assume last layer is softmax dense)
        if len(model.layers) >= 2:
            feature_output = model.layers[-2].output
        else:
            raise ValueError("Could not identify a feature layer in model.")

    feature_extractor = Model(inputs=model.input, outputs=feature_output)
    return feature_extractor

# -------------------------
# Load pretrained models and build extractors
# -------------------------
print("Loading pretrained models...")
emotion_model = load_model(EMOTION_MODEL_PATH)
pos_model = load_model(POS_MODEL_PATH)
print("Models loaded.")

print("Creating feature extractors...")
emotion_extractor = get_feature_extractor(emotion_model)
pos_extractor = get_feature_extractor(pos_model)

# Freeze
emotion_extractor.trainable = False
pos_extractor.trainable = False

# Determine feature dimensions by running a dummy input through extractors
dummy = np.zeros((1, ) + INPUT_SHAPE, dtype=np.float32)
feat_a = emotion_extractor.predict(dummy)
feat_b = pos_extractor.predict(dummy)
feat_dim_a = int(np.prod(feat_a.shape[1:]))
feat_dim_b = int(np.prod(feat_b.shape[1:]))
print(f"Feature dims -> emotion: {feat_dim_a}, pos: {feat_dim_b}")

# -------------------------
# Build fusion model - FIXED APPROACH
# -------------------------
# Create the main input
main_input = Input(shape=INPUT_SHAPE, name="eeg_input")

# Run input through extractors
feat_a_tensor = emotion_extractor(main_input)  # shape: (batch, ...feat_a_shape)
feat_b_tensor = pos_extractor(main_input)      # shape: (batch, ...feat_b_shape)

# Flatten both to vectors
feat_a_flat = Flatten(name="emotion_features_flat")(feat_a_tensor)
feat_b_flat = Flatten(name="pos_features_flat")(feat_b_tensor)

# Concatenate
concat = Concatenate(name="feature_concat")([feat_a_flat, feat_b_flat])

# Build fusion layers directly (no intermediate model)
x = Dense(FUSION_DENSE, activation="relu", name="fusion_dense")(concat)
x = BatchNormalization(name="fusion_bn")(x)
x = Dropout(0.3, name="fusion_dropout")(x)

# Two output heads
emotion_out = Dense(NUM_CLASSES_EMOTION, activation="softmax", name="emotion")(x)
pos_out = Dense(NUM_CLASSES_POS, activation="softmax", name="pos")(x)

# Final combined model
combined_model = Model(inputs=main_input, outputs=[emotion_out, pos_out], name="combined_multihead_model")
print(combined_model.summary())

# -------------------------
# Load datasets and build combined training arrays
# -------------------------
print("Loading datasets...")
emotion_data = load_pickle(EMOTION_DATA_PATH)  # expect dict with "X","y"
pos_data = load_pickle(POS_DATA_PATH)

X_e = np.array(emotion_data["data"])  # shape (n_e, 4, 1000)
X_e = X_e.reshape(-1, 4, 60)
channel_weights = np.array([1.25, 1.25, 0.9, 0.9]).reshape(1, 4, 1)
X_e = X_e * channel_weights
X_e = (X_e - X_e.mean(axis=2, keepdims=True)) / X_e.std(axis=2, keepdims=True)
y_e = np.array(emotion_data["labels"])
# def balanced_oversample(X, y):
#     classes, counts = np.unique(y, return_counts=True)
#     max_count = counts.max()
#     indices_list = []

#     for cls in classes:
#         cls_indices = np.where(y == cls)[0]
#         if len(cls_indices) < max_count:
#             sampled_indices = np.random.choice(cls_indices, max_count, replace=True)
#         else:
#             sampled_indices = cls_indices
#         indices_list.append(sampled_indices)

#     balanced_indices = np.concatenate(indices_list)
#     np.random.shuffle(balanced_indices)

#     return X[balanced_indices], y[balanced_indices]

# # Usage:
# X_e, y_e = balanced_oversample(X_e, y_e)

X_p = np.array(pos_data["data"]) 
X_p = X_p.reshape(-1, 4, 60)     # shape (n_p, 4, 60)
X_p = (X_p - X_p.mean(axis=2, keepdims=True)) / X_p.std(axis=2, keepdims=True)
y_p = np.array(pos_data["labels"])
y_p = y_p - 1

# Check shapes
print("Shapes:", X_e.shape, y_e.shape, X_p.shape, y_p.shape)

# Reshape inputs to add channel dim for Keras 
X_e_pre = X_e[..., np.newaxis] 
X_p_pre = X_p[..., np.newaxis] 

# Build combined dataset by concatenation of examples (we'll use sample weights)
X_combined = np.concatenate([X_e_pre, X_p_pre], axis=0)

# Labels: Build placeholders with default values (won't be used where mask=0)
y_emotion_combined = np.concatenate([
    to_categorical(y_e, NUM_CLASSES_EMOTION),
    np.zeros((len(X_p_pre), NUM_CLASSES_EMOTION))
], axis=0)

y_pos_combined = np.concatenate([
    np.zeros((len(X_e_pre), NUM_CLASSES_POS)),
    to_categorical(y_p, NUM_CLASSES_POS)
], axis=0)

# Sample weights: 1 where label exists, 0 where missing
sw_emotion = np.concatenate([
    np.ones(len(X_e_pre)),
    np.zeros(len(X_p_pre))
], axis=0)

sw_pos = np.concatenate([
    np.zeros(len(X_e_pre)),
    np.ones(len(X_p_pre))
], axis=0)

print("Combined shapes:", X_combined.shape, y_emotion_combined.shape, y_pos_combined.shape)
print("Sample weights:", sw_emotion.sum(), sw_pos.sum())

# -------------------------
# Train/validation split
# -------------------------
X_train, X_val, ye_train, ye_val, yp_train, yp_val, swe_train, swe_val, swp_train, swp_val = train_test_split(
    X_combined, y_emotion_combined, y_pos_combined, sw_emotion, sw_pos,
    test_size=0.2, random_state=RANDOM_STATE, shuffle=True
)
def add_gaussian_noise(inputs, noise_std=0.01):
    noise = tf.random.normal(shape=tf.shape(inputs), mean=0.0, stddev=noise_std)
    return inputs + noise

# Example: inside your training data pipeline or model input:
X_train_= add_gaussian_noise(X_train, noise_std=0.02)
# -------------------------
# Compile & train
# -------------------------
# Debug: Print layer names and output names
print("Layer names:")
for i, layer in enumerate(combined_model.layers):
    print(f"  {i}: {layer.name} ({type(layer).__name__})")

print("Output tensor names:")
for i, output in enumerate(combined_model.outputs):
    print(f"  {i}: {output.name}")

from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.class_weight import compute_class_weight

# ---- Emotion ----
emotion_labels = np.argmax(ye_train, axis=1)
classes_emotion = np.unique(emotion_labels[swe_train > 0])  # only where mask=1
class_weights_emotion = compute_class_weight(
    class_weight='balanced',
    classes=classes_emotion,
    y=emotion_labels[swe_train > 0]
)
class_weights_emotion_dict = dict(zip(classes_emotion, class_weights_emotion))

# Map each training sample to its weight
emotion_balanced_sw = np.array([
    class_weights_emotion_dict[label] if mask == 1 else 0
    for label, mask in zip(emotion_labels, swe_train)
], dtype=np.float32)

# ---- POS ----
pos_labels = np.argmax(yp_train, axis=1)
classes_pos = np.unique(pos_labels[swp_train > 0])
class_weights_pos = compute_class_weight(
    class_weight='balanced',
    classes=classes_pos,
    y=pos_labels[swp_train > 0]
)
class_weights_pos_dict = dict(zip(classes_pos, class_weights_pos))

pos_balanced_sw = np.array([
    class_weights_pos_dict[label] if mask == 1 else 0
    for label, mask in zip(pos_labels, swp_train)
], dtype=np.float32)

print(pos_balanced_sw, emotion_balanced_sw)

# Try using list format instead of dictionary format for outputs
combined_model.compile(
    optimizer=Adam(learning_rate=LR),
    loss=["categorical_crossentropy", "categorical_crossentropy"],
    metrics=["accuracy", "accuracy"],
    loss_weights=[1.0, 1.0]  # Equal weighting
)

print("Starting training...")


# Use list format for targets and sample weights
combined_model.fit(
    X_train,
    [ye_train, yp_train],
    sample_weight=[emotion_balanced_sw, pos_balanced_sw],  # now includes balancing
    validation_data=(X_val, [ye_val, yp_val], [swe_val, swp_val]),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)


# -------------------------
# Save combined model
# -------------------------
combined_model.save("multi_head_fusion_model_with_proper_class_weights_and_noise_with_channel_weighting.h5")
print("Saved multi-head fusion model to multi_head_fusion_model.h5")
model = load_model("multi_head_fusion_model_with_proper_class_weights_and_noise_with_channel_weighting.h5")
# Evaluate on validation set
val_results = combined_model.evaluate(
    X_val, [ye_val, yp_val], 
    sample_weight=[swe_val, swp_val],
    verbose=1
)

print(f"\n✅ Validation Results:")
print(f"Total Loss: {val_results[0]:.4f}")
print(f"Emotion Loss: {val_results[1]:.4f}")  
print(f"POS Loss: {val_results[2]:.4f}")
print(f"Emotion Accuracy: {val_results[3]:.4f}")
print(f"POS Accuracy: {val_results[4]:.4f}")
print(combined_model.summary())

# -------------------------
# Inference helper
# -------------------------
# def predict_on_trial(model, trial):
#     """trial shape expected (4,60) or (4,60,1)"""
#     t = np.array(trial)
#     if t.ndim == 2:
#         t = t[..., np.newaxis]
#     t = np.expand_dims(t, axis=0).astype("float32")
#     emotion_prob, pos_prob = model.predict(t, verbose=0)
#     return emotion_prob[0], pos_prob[0]

# Example usage:
# print("\n--- Example Inference ---")
# trial = X_combined[0]
# eprob, pprob = predict_on_trial(combined_model, trial)
# print("Emotion probabilities:", eprob)
# print("POS probabilities:", pprob)
# print("Predicted emotion class:", np.argmax(eprob))
# print("Predicted POS class:", np.argmax(pprob))


import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
ye_val_labels = np.argmax(ye_val, axis=1)
yp_val_labels = np.argmax(yp_val, axis=1)

# --- 1. Predict on the validation set ---
y_pred_emotion, y_pred_pos = model.predict(X_val)

# Convert from probabilities to class labels
y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
y_pred_pos_classes = np.argmax(y_pred_pos, axis=1)

# --- 2. Compute confusion matrices ---
cm_emotion = confusion_matrix(ye_val_labels, y_pred_emotion_classes)
cm_pos = confusion_matrix(yp_val_labels, y_pred_pos_classes)

# --- 3. Plot confusion matrices ---
from sklearn.metrics import accuracy_score
acc_emotion = accuracy_score(ye_val_labels, y_pred_emotion_classes)
acc_pos = accuracy_score(yp_val_labels, y_pred_pos_classes)

# Plot confusion matrices with accuracies in title
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

disp_emotion = ConfusionMatrixDisplay(confusion_matrix=cm_emotion)
disp_emotion.plot(ax=axes[0], cmap=plt.cm.Blues, colorbar=False)
axes[0].set_title(f"Emotion Confusion Matrix\nAccuracy: {acc_emotion:.2%}")

disp_pos = ConfusionMatrixDisplay(confusion_matrix=cm_pos)
disp_pos.plot(ax=axes[1], cmap=plt.cm.Oranges, colorbar=False)
axes[1].set_title(f"POS Confusion Matrix\nAccuracy: {acc_pos:.2%}")

plt.tight_layout()
plt.show()
