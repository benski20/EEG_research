# multi_task_fusion.py
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization
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

def make_multi_head_model(feat_dim_a, feat_dim_b):
    # Input
    inp = Input(shape=INPUT_SHAPE, name="eeg_input")

    # Placeholder tensors for the extractors will be connected later using functional API
    # Build a small fusion network that accepts concatenated features
    concat_inp = Input(shape=(feat_dim_a + feat_dim_b,), name="concat_input")

    x = Dense(FUSION_DENSE, activation="relu")(concat_inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    # Two heads
    emotion_out = Dense(NUM_CLASSES_EMOTION, activation="softmax", name="emotion")(x)
    pos_out = Dense(NUM_CLASSES_POS, activation="softmax", name="pos")(x)

    fusion_model = Model(inputs=concat_inp, outputs=[emotion_out, pos_out], name="fusion_model")
    return inp, fusion_model

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
# Build fusion model graph
# -------------------------
# Get placeholders and fusion head
inp_placeholder, fusion_head = make_multi_head_model(feat_dim_a, feat_dim_b)

# We'll create the final model by:
#  - mapping inp_placeholder through both extractors
#  - flattening their outputs and concatenating
#  - passing concatenation to fusion_head
from tensorflow.keras.layers import Flatten, Reshape

# Run input through extractors
feat_a_tensor = emotion_extractor(inp_placeholder)  # shape: (batch, ...feat_a_shape)
feat_b_tensor = pos_extractor(inp_placeholder)      # shape: (batch, ...feat_b_shape)

# Flatten both to vectors
feat_a_flat = Flatten()(feat_a_tensor)
feat_b_flat = Flatten()(feat_b_tensor)

# Concatenate
concat = Concatenate()( [feat_a_flat, feat_b_flat] )

# Pass through fusion head layers manually (reuse fusion_head layers)
# To keep things simple, reconstruct same operations: Dense -> BN -> Dropout -> heads
x = fusion_head.layers[1](concat)  # Dense(FUSION_DENSE)
x = fusion_head.layers[2](x)       # BatchNorm
x = fusion_head.layers[3](x)       # Dropout
emotion_out = fusion_head.layers[4](x)  # Dense(NUM_CLASSES_EMOTION)
pos_out = fusion_head.layers[5](x)      # Dense(NUM_CLASSES_POS)

# Final combined model
combined_model = Model(inputs=inp_placeholder, outputs=[emotion_out, pos_out], name="combined_multihead_model")
print(combined_model.summary())

# -------------------------
# Load datasets and build combined training arrays
# -------------------------
print("Loading datasets...")
emotion_data = load_pickle(EMOTION_DATA_PATH)  # expect dict with "X","y"
pos_data = load_pickle(POS_DATA_PATH)

X_e = np.array(emotion_data["data"])  # shape (n_e, 4, 1000)
X_e = X_e.reshape(-1, 4, 60)
X_e = (X_e - X_e.mean(axis=2, keepdims=True)) / X_e.std(axis=2, keepdims=True)
y_e = np.array(emotion_data["labels"])


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

# -------------------------
# Compile & train
# -------------------------

combined_model.compile(
    optimizer=Adam(learning_rate=LR),
    loss={"emotion": "categorical_crossentropy", "pos": "categorical_crossentropy"},
    metrics={"emotion": "accuracy", "pos": "accuracy"}
)

print("Starting training...")
print([out.name for out in combined_model.outputs])

combined_model.fit(
    X_train,
    {"emotion": ye_train, "pos": yp_train},
    sample_weight={"emotion": swe_train, "pos": swp_train},
    validation_data=(X_val, {"emotion": ye_val, "pos": yp_val}, {"emotion": swe_val, "pos": swp_val}),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# -------------------------
# Save combined model
# -------------------------
combined_model.save("multi_head_fusion_model.h5")
print("Saved multi-head fusion model to multi_head_fusion_model.h5")
print(f"\n✅ Test Accuracy: {combined_model.metrics:.4f}")
print(combined_model.summary())

# -------------------------
# Inference helper
# -------------------------
# def predict_on_trial(model, trial):
#     """trial shape expected (4,1000) or (4,1000,1)"""
#     t = np.array(trial)
#     if t.ndim == 2:
#         t = t[..., np.newaxis]
#     t = np.expand_dims(t, axis=0).astype("float32")
#     emotion_prob, pos_prob = model.predict(t)
#     return emotion_prob[0], pos_prob[0]

# Example usage:
# trial = X_combined[0]
# eprob, pprob = predict_on_trial(combined_model, trial)
# print("Emotion probs:", eprob, "POS probs:", pprob)

