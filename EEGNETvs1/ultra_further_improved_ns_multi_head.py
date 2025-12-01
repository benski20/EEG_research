# improved_multi_task_fusion_balanced_ns_nopriors.py - BALANCED + NEUROSYMBOLIC FUSION (NO PRIORS)
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# -------------------------
# CONFIG
# -------------------------
EMOTION_MODEL_PATH = "eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5"
POS_MODEL_PATH = "eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5"
EMOTION_DATA_PATH = "eeg_fft_dataset.pkl"
POS_DATA_PATH = "eeg_fft_nounvsverb_JASON_dataset.pkl"

INPUT_SHAPE = (4, 60, 1)
NUM_CLASSES_EMOTION = 3
NUM_CLASSES_POS = 2
FUSION_DENSE = 256
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
RANDOM_STATE = 42

# -------------------------
# BALANCED FOCAL LOSS
# -------------------------
def balanced_focal_loss(alpha=0.25, gamma=1.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        ce = -y_true * tf.math.log(y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))
    return focal_loss_fixed

# -------------------------
# NEUROSYMBOLIC LOSS (no priors)
# -------------------------
def neurosymbolic_loss(y_true_emotion, y_pred_emotion, y_true_pos, y_pred_pos, alpha=0.1):
    """
    Encourages consistency between emotion and POS heads using shared features.
    Simple dot-product consistency between predicted distributions.
    """
    # Cosine similarity-like regularization
    y_pred_emotion_norm = tf.nn.l2_normalize(y_pred_emotion, axis=-1)
    y_pred_pos_norm = tf.nn.l2_normalize(y_pred_pos, axis=-1)
    similarity = tf.reduce_sum(y_pred_emotion_norm * y_pred_pos_norm, axis=-1)
    return alpha * tf.reduce_mean(1.0 - similarity)

def combined_loss(y_true_emotion, y_pred_emotion, y_true_pos, y_pred_pos):
    base_emotion_loss = balanced_focal_loss(alpha=0.25, gamma=1.5)(y_true_emotion, y_pred_emotion)
    base_pos_loss = balanced_focal_loss(alpha=0.25, gamma=1.5)(y_true_pos, y_pred_pos)
    ns_loss = neurosymbolic_loss(y_true_emotion, y_pred_emotion, y_true_pos, y_pred_pos, alpha=0.1)
    return base_emotion_loss + base_pos_loss + ns_loss

# -------------------------
# DATA IMPROVEMENTS
# -------------------------
def gentle_emotion_preprocessing(X_emotion):
    X_processed = X_emotion.copy().astype(np.float32)
    for ch in range(X_processed.shape[1]):
        channel_data = X_processed[:, ch, :]
        p25 = np.percentile(channel_data, 25, axis=1, keepdims=True)
        p75 = np.percentile(channel_data, 75, axis=1, keepdims=True)
        scale = np.where(p75 - p25 == 0, 1, p75 - p25)
        X_processed[:, ch, :] = (channel_data - p25) / scale
    return X_processed

def conservative_neutral_augmentation(X_emotion, y_emotion, neutral_class=1):
    neutral_mask = (y_emotion == neutral_class)
    neutral_samples = X_emotion[neutral_mask]
    augmented_samples = [X_emotion]
    augmented_labels = [y_emotion]
    neutral_augmented = []
    for sample in neutral_samples:
        aug_sample = sample.copy()
        noise_std = 0.01 * np.std(aug_sample)
        aug_sample += np.random.normal(0, noise_std, aug_sample.shape)
        neutral_augmented.append(aug_sample)
    if len(neutral_augmented) > 0:
        augmented_samples.append(np.array(neutral_augmented))
        augmented_labels.append(np.full(len(neutral_augmented), neutral_class))
    X_final = np.concatenate(augmented_samples, axis=0)
    y_final = np.concatenate(augmented_labels, axis=0)
    return X_final, y_final

def conservative_class_weights(y_emotion):
    classes = np.unique(y_emotion)
    standard_weights = compute_class_weight('balanced', classes=classes, y=y_emotion)
    emotion_weights = dict(zip(classes, standard_weights))
    if 1 in emotion_weights:
        emotion_weights[1] *= 1.3
    return emotion_weights

def robust_normalize_eeg(X):
    for ch in range(X.shape[1]):
        channel_data = X[:, ch, :]
        p5 = np.percentile(channel_data, 5, axis=1, keepdims=True)
        p95 = np.percentile(channel_data, 95, axis=1, keepdims=True)
        scale = np.where(p95 - p5 == 0, 1, p95 - p5)
        X[:, ch, :] = (channel_data - p5) / scale
    return X

def get_feature_extractor(model):
    flatten_layer = None
    for layer in model.layers:
        if layer.__class__.__name__.lower() == "flatten" or "flatten" in layer.name:
            flatten_layer = layer
            break
    if flatten_layer is not None:
        feature_output = flatten_layer.output
    else:
        feature_output = model.layers[-2].output
    feature_extractor = Model(inputs=model.input, outputs=feature_output)
    return feature_extractor

def create_simple_attention_fusion(emotion_features, pos_features, fusion_dim=256):
    emotion_proj = Dense(fusion_dim//2, activation='relu')(emotion_features)
    pos_proj = Dense(fusion_dim//2, activation='relu')(pos_features)
    emotion_attention = Dense(1, activation='sigmoid')(emotion_proj)
    pos_attention = Dense(1, activation='sigmoid')(pos_proj)
    emotion_weighted = Lambda(lambda x: x[0] * x[1])([emotion_proj, emotion_attention])
    pos_weighted = Lambda(lambda x: x[0] * x[1])([pos_proj, pos_attention])
    fused = Concatenate()([emotion_weighted, pos_weighted])
    return fused

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def create_stratified_split(X, y_emotion, y_pos, test_size=0.2, random_state=42):
    combined_labels = []
    for i in range(len(X)):
        if i < len(y_emotion):
            combined_labels.append(f"emotion_{np.argmax(y_emotion[i])}")
        else:
            combined_labels.append(f"pos_{np.argmax(y_pos[i-len(y_emotion)])}")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(sss.split(X, combined_labels))
    return train_idx, val_idx
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

def evaluate_and_visualize(model, X_val, ye_val, yp_val, sw_emotion, sw_pos):
    # Predict
    y_pred_emotion, y_pred_pos = model.predict(X_val)
    y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
    y_pred_pos_classes = np.argmax(y_pred_pos, axis=1)
    
    ye_val_classes = np.argmax(ye_val, axis=1)
    yp_val_classes = np.argmax(yp_val, axis=1)
    
    # Masks
    emotion_mask = sw_emotion > 0
    pos_mask = sw_pos > 0
    
    # --- Emotion Evaluation ---
    print("\n🎭 EMOTION CLASSIFICATION RESULTS:")
    print(classification_report(
        ye_val_classes[emotion_mask],
        y_pred_emotion_classes[emotion_mask],
        target_names=["Negative", "Neutral", "Positive"]
    ))
    
    cm_emotion = confusion_matrix(ye_val_classes[emotion_mask], y_pred_emotion_classes[emotion_mask])
    disp_emotion = ConfusionMatrixDisplay(cm_emotion, display_labels=["Negative", "Neutral", "Positive"])
    disp_emotion.plot(cmap=plt.cm.Blues)
    plt.title("Emotion Confusion Matrix")
    plt.show()
    
    # --- POS Evaluation ---
    print("\n📝 POS CLASSIFICATION RESULTS:")
    print(classification_report(
        yp_val_classes[pos_mask],
        y_pred_pos_classes[pos_mask],
        target_names=["Noun", "Verb"]
    ))
    
    cm_pos = confusion_matrix(yp_val_classes[pos_mask], y_pred_pos_classes[pos_mask])
    disp_pos = ConfusionMatrixDisplay(cm_pos, display_labels=["Noun", "Verb"])
    disp_pos.plot(cmap=plt.cm.Greens)
    plt.title("POS Confusion Matrix")
    plt.show()
    
    # Optional: Per-class accuracy bars
    emotion_acc = np.diag(cm_emotion) / np.sum(cm_emotion, axis=1) * 100
    pos_acc = np.diag(cm_pos) / np.sum(cm_pos, axis=1) * 100
    
    plt.bar(["Neg", "Neu", "Pos"], emotion_acc, color='skyblue')
    plt.title("Emotion Per-Class Accuracy (%)")
    plt.ylabel("Accuracy")
    plt.show()
    
    plt.bar(["Noun", "Verb"], pos_acc, color='lightgreen')
    plt.title("POS Per-Class Accuracy (%)")
    plt.ylabel("Accuracy")
    plt.show()

def apply_conservative_emotion_improvements(X_emotion, y_emotion):
    X_processed = gentle_emotion_preprocessing(X_emotion)
    X_augmented, y_augmented = conservative_neutral_augmentation(X_processed, y_emotion)
    class_weights = conservative_class_weights(y_augmented)
    return X_augmented, y_augmented, class_weights

# -------------------------
# MAIN EXECUTION
# -------------------------
def main():
    # Load pretrained models
    emotion_model = load_model(EMOTION_MODEL_PATH)
    pos_model = load_model(POS_MODEL_PATH)
    emotion_extractor = get_feature_extractor(emotion_model)
    pos_extractor = get_feature_extractor(pos_model)
    emotion_extractor.trainable = False
    pos_extractor.trainable = False

    # Dummy forward to get feature dims
    dummy = np.zeros((1,) + INPUT_SHAPE, dtype=np.float32)

    main_input = Input(shape=INPUT_SHAPE)
    feat_a_flat = Flatten()(emotion_extractor(main_input))
    feat_b_flat = Flatten()(pos_extractor(main_input))
    fused_features = create_simple_attention_fusion(feat_a_flat, feat_b_flat, fusion_dim=FUSION_DENSE)
    x = Dense(FUSION_DENSE, activation="relu")(fused_features)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(FUSION_DENSE//2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    emotion_out = Dense(NUM_CLASSES_EMOTION, activation="softmax", name="emotion")(x)
    pos_out = Dense(NUM_CLASSES_POS, activation="softmax", name="pos")(x)

    combined_model = Model(inputs=main_input, outputs=[emotion_out, pos_out])

    # Load datasets
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data = load_pickle(POS_DATA_PATH)

    X_e = np.array(emotion_data["data"]).reshape(-1, 4, 60)
    y_e = np.array(emotion_data["labels"])
    X_e_improved, y_e_improved, _ = apply_conservative_emotion_improvements(X_e, y_e)
    y_e_improved_cat = to_categorical(y_e_improved, NUM_CLASSES_EMOTION)

    X_p = np.array(pos_data["data"]).reshape(-1, 4, 60)
    X_p = robust_normalize_eeg(X_p)
    y_p = np.array(pos_data["labels"]) - 1
    y_p_cat = to_categorical(y_p, NUM_CLASSES_POS)

    # Combine datasets
    X_combined = np.concatenate([X_e_improved[..., np.newaxis], X_p[..., np.newaxis]], axis=0)
    y_emotion_combined = np.concatenate([y_e_improved_cat, np.zeros((len(X_p), NUM_CLASSES_EMOTION))], axis=0)
    y_pos_combined = np.concatenate([np.zeros((len(X_e_improved), NUM_CLASSES_POS)), y_p_cat], axis=0)

    sw_emotion = np.concatenate([np.ones(len(X_e_improved)), np.zeros(len(X_p))])
    sw_pos = np.concatenate([np.zeros(len(X_e_improved)), np.ones(len(X_p))])

    train_idx, val_idx = create_stratified_split(X_combined, y_emotion_combined, y_pos_combined)
    X_train, X_val = X_combined[train_idx], X_combined[val_idx]
    ye_train, ye_val = y_emotion_combined[train_idx], y_emotion_combined[val_idx]
    yp_train, yp_val = y_pos_combined[train_idx], y_pos_combined[val_idx]
    swe_train, swe_val = sw_emotion[train_idx], sw_emotion[val_idx]
    swp_train, swp_val = sw_pos[train_idx], sw_pos[val_idx]

    # # Compile model
    # combined_model.compile(
    #     optimizer=Adam(learning_rate=LR),
    #     loss=lambda yt, yp: combined_loss(yt[0], yp[0], yt[1], yp[1]),
    #     metrics=[['accuracy'], ['accuracy']]
    # )

    # callbacks = [
    #     EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    #     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7),
    #     ModelCheckpoint(filepath='best_balanced_ns_fusion_model_nopriors.h5', monitor='val_loss', save_best_only=True)
    # ]

    # history = combined_model.fit(
    #     X_train, [ye_train, yp_train],
    #     sample_weight=[swe_train, swp_train],
    #     validation_data=(X_val, [ye_val, yp_val], [swe_val, swp_val]),
    #     epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks, verbose=1
    # )

    model = load_model("best_balanced_ns_fusion_model_nopriors.h5")
    evaluate_and_visualize(model, X_val, ye_val, yp_val, swe_val, swp_val)

    print("✅ Training completed!")
    return combined_model, history

if __name__ == "__main__":
    model, history = main()
