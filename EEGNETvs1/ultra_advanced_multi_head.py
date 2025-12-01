# improved_multi_task_fusion_balanced.py - BALANCED APPROACH FOR EMOTION IMPROVEMENT
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten,
    MultiHeadAttention, LayerNormalization, Add, Lambda, Reshape
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# -------------------------
# BALANCED CONFIG - CONSERVATIVE IMPROVEMENTS
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
LR = 1e-4           # Back to reasonable learning rate
RANDOM_STATE = 42

# -------------------------
# GENTLE FOCAL LOSS - BALANCED APPROACH
# -------------------------
def balanced_focal_loss(alpha=0.25, gamma=1.5):
    """
    Gentle focal loss - less aggressive than before
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        ce = -y_true * tf.math.log(y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        focal_loss = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    
    return focal_loss_fixed

# -------------------------
# CONSERVATIVE DATA IMPROVEMENTS
# -------------------------
def gentle_emotion_preprocessing(X_emotion):
    """
    Light preprocessing - just basic improvements
    """
    X_processed = X_emotion.copy().astype(np.float32)
    
    # Simple robust normalization per channel
    for ch in range(X_processed.shape[1]):
        channel_data = X_processed[:, ch, :]
        p25 = np.percentile(channel_data, 25, axis=1, keepdims=True)
        p75 = np.percentile(channel_data, 75, axis=1, keepdims=True)
        
        scale = p75 - p25
        scale = np.where(scale == 0, 1, scale)
        X_processed[:, ch, :] = (channel_data - p25) / scale
    
    return X_processed

def conservative_neutral_augmentation(X_emotion, y_emotion, neutral_class=1, multiplier=1):
    """
    Conservative augmentation - just 1x extra neutral samples with light modifications
    """
    neutral_mask = (y_emotion == neutral_class)
    neutral_samples = X_emotion[neutral_mask]
    
    print(f"🎭 Original neutral samples: {np.sum(neutral_mask)}")
    
    if len(neutral_samples) == 0:
        return X_emotion, y_emotion
    
    # Keep all original samples
    augmented_samples = [X_emotion]
    augmented_labels = [y_emotion]
    
    # Add only small amount of augmented neutral samples
    neutral_augmented = []
    for sample in neutral_samples:
        # Very light augmentation
        aug_sample = sample.copy()
        
        # Light noise only
        noise_std = 0.01 * np.std(aug_sample)  # Very small noise
        noise = np.random.normal(0, noise_std, aug_sample.shape)
        aug_sample += noise
        
        neutral_augmented.append(aug_sample)
    
    # Add augmented neutral samples
    if len(neutral_augmented) > 0:
        augmented_samples.append(np.array(neutral_augmented))
        augmented_labels.append(np.full(len(neutral_augmented), neutral_class))
    
    X_final = np.concatenate(augmented_samples, axis=0)
    y_final = np.concatenate(augmented_labels, axis=0)
    
    print(f"🎭 After conservative augmentation: {np.bincount(y_final)}")
    
    return X_final, y_final

def conservative_class_weights(y_emotion):
    """
    Mild class weight adjustment - not too aggressive
    """
    classes = np.unique(y_emotion)
    standard_weights = compute_class_weight('balanced', classes=classes, y=y_emotion)
    
    emotion_weights = dict(zip(classes, standard_weights))
    
    # Only small boost for neutral class
    if 1 in emotion_weights:
        emotion_weights[1] *= 1.3  # Gentle boost instead of 2.0
    
    print(f"🎯 Conservative class weights: {emotion_weights}")
    return emotion_weights

# -------------------------
# SIMPLE FUSION ARCHITECTURE
# -------------------------
def create_simple_attention_fusion(emotion_features, pos_features, fusion_dim=256):
    """
    Simple attention fusion - not overly complex
    """
    # Project both to same dimension
    emotion_proj = Dense(fusion_dim//2, activation='relu', name='emotion_proj')(emotion_features)
    pos_proj = Dense(fusion_dim//2, activation='relu', name='pos_proj')(pos_features)
    
    # Simple attention weights
    emotion_attention = Dense(1, activation='sigmoid', name='emotion_att')(emotion_proj)
    pos_attention = Dense(1, activation='sigmoid', name='pos_att')(pos_proj)
    
    # Apply attention
    emotion_weighted = Lambda(lambda x: x[0] * x[1])([emotion_proj, emotion_attention])
    pos_weighted = Lambda(lambda x: x[0] * x[1])([pos_proj, pos_attention])
    
    # Simple concatenation
    fused = Concatenate()([emotion_weighted, pos_weighted])
    
    return fused

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def get_feature_extractor(model):
    """Extract features before final classification layer"""
    flatten_layer = None
    for layer in model.layers:
        if layer.__class__.__name__.lower() == "flatten" or "flatten" in layer.name:
            flatten_layer = layer
            break
    
    if flatten_layer is not None:
        feature_output = flatten_layer.output
    else:
        if len(model.layers) >= 2:
            feature_output = model.layers[-2].output
        else:
            raise ValueError("Could not identify a feature layer in model.")
    
    feature_extractor = Model(inputs=model.input, outputs=feature_output)
    return feature_extractor

def robust_normalize_eeg(X):
    """Standard robust normalization"""
    for ch in range(X.shape[1]):
        channel_data = X[:, ch, :]
        p5 = np.percentile(channel_data, 5, axis=1, keepdims=True)
        p95 = np.percentile(channel_data, 95, axis=1, keepdims=True)
        
        scale = p95 - p5
        scale = np.where(scale == 0, 1, scale)
        X[:, ch, :] = (channel_data - p5) / scale
    
    return X

def create_stratified_split(X, y_emotion, y_pos, test_size=0.2, random_state=42):
    """Create stratified split considering both tasks"""
    combined_labels = []
    for i in range(len(X)):
        if i < len(y_emotion):
            combined_labels.append(f"emotion_{y_emotion[i]}")
        else:
            combined_labels.append(f"pos_{y_pos[i-len(y_emotion)]}")
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(sss.split(X, combined_labels))
    
    return train_idx, val_idx


# -------------------------
# CONSERVATIVE EMOTION IMPROVEMENTS
# -------------------------
def apply_conservative_emotion_improvements(X_emotion, y_emotion):
    """
    Apply gentle improvements - no over-correction
    """
    print("🎯 Applying conservative emotion improvements...")
    
    # 1. Light preprocessing
    print("🔧 Light preprocessing...")
    X_processed = gentle_emotion_preprocessing(X_emotion)
    
    # 2. Conservative neutral augmentation
    print("🎭 Conservative neutral augmentation...")
    X_augmented, y_augmented = conservative_neutral_augmentation(X_processed, y_emotion)
    
    # 3. Standard SMOTE with conservative parameters
    print("⚖️ Conservative SMOTE...")
    original_shape = X_augmented.shape
    X_flat = X_augmented.reshape(X_augmented.shape[0], -1)
    
    # Only apply SMOTE if there's significant imbalance
    class_counts = np.bincount(y_augmented)
    min_count = np.min(class_counts)
    max_count = np.max(class_counts)
    
    if max_count > min_count * 1.5:  # Only if >50% imbalance
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_flat, y_augmented = smote.fit_resample(X_flat, y_augmented)
        X_balanced = X_flat.reshape(-1, *original_shape[1:])
    else:
        print("📊 Classes reasonably balanced, skipping SMOTE")
        X_balanced = X_augmented
    
    # 4. Conservative class weights
    class_weights = conservative_class_weights(y_augmented)
    
    print(f"📊 Final emotion dataset shape: {X_balanced.shape}")
    print(f"📊 Final emotion distribution: {np.bincount(y_augmented)}")
    
    return X_balanced, y_augmented, class_weights

# -------------------------
# MAIN EXECUTION
# -------------------------
def main():
    print("🚀 Loading pretrained models...")
    emotion_model = load_model(EMOTION_MODEL_PATH)
    pos_model = load_model(POS_MODEL_PATH)
    
    print("🔧 Creating feature extractors...")
    emotion_extractor = get_feature_extractor(emotion_model)
    pos_extractor = get_feature_extractor(pos_model)
    
    # Freeze extractors
    emotion_extractor.trainable = False
    pos_extractor.trainable = False
    
    # Get feature dimensions
    dummy = np.zeros((1,) + INPUT_SHAPE, dtype=np.float32)
    feat_a = emotion_extractor.predict(dummy, verbose=0)
    feat_b = pos_extractor.predict(dummy, verbose=0)
    feat_dim_a = int(np.prod(feat_a.shape[1:]))
    feat_dim_b = int(np.prod(feat_b.shape[1:]))
    
    print(f"📊 Feature dimensions -> Emotion: {feat_dim_a}, POS: {feat_dim_b}")
    
    # -------------------------
    # SIMPLE MODEL ARCHITECTURE
    # -------------------------
    print("🏗️ Building balanced fusion model...")
    main_input = Input(shape=INPUT_SHAPE, name="eeg_input")
    
    # Feature extraction
    feat_a_tensor = emotion_extractor(main_input)
    feat_b_tensor = pos_extractor(main_input)
    
    feat_a_flat = Flatten(name="emotion_features_flat")(feat_a_tensor)
    feat_b_flat = Flatten(name="pos_features_flat")(feat_b_tensor)
    
    # Simple attention fusion
    fused_features = create_simple_attention_fusion(feat_a_flat, feat_b_flat, fusion_dim=FUSION_DENSE)
    
    # Standard fusion layers
    x = Dense(FUSION_DENSE, activation="relu", name="fusion_dense1")(fused_features)
    x = BatchNormalization(name="fusion_bn1")(x)
    x = Dropout(0.3, name="fusion_dropout1")(x)  # Reduced dropout
    
    x = Dense(FUSION_DENSE//2, activation="relu", name="fusion_dense2")(x)
    x = BatchNormalization(name="fusion_bn2")(x)
    x = Dropout(0.2, name="fusion_dropout2")(x)
    
    # Simple output heads - no special treatment
    emotion_out = Dense(NUM_CLASSES_EMOTION, activation="softmax", name="emotion")(x)
    pos_out = Dense(NUM_CLASSES_POS, activation="softmax", name="pos")(x)
    
    combined_model = Model(inputs=main_input, outputs=[emotion_out, pos_out], 
                          name="balanced_fusion_model")
    
    # -------------------------
    # DATA LOADING AND PREPROCESSING
    # -------------------------
    print("📂 Loading datasets...")
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data = load_pickle(POS_DATA_PATH)
    
    # Process emotion data with conservative improvements
    X_e = np.array(emotion_data["data"]).reshape(-1, 4, 60)
    y_e = np.array(emotion_data["labels"])
    
    # Apply conservative improvements
    X_e_improved, y_e_improved, emotion_class_weights = apply_conservative_emotion_improvements(X_e, y_e)
    
    # Process POS data normally
    X_p = np.array(pos_data["data"]).reshape(-1, 4, 60)
    X_p = robust_normalize_eeg(X_p)
    y_p = np.array(pos_data["labels"]) - 1
    
    # Light SMOTE for POS if needed
    X_p_balanced, y_p_balanced = X_p, y_p
    pos_counts = np.bincount(y_p)
    if len(pos_counts) > 1 and np.max(pos_counts) > np.min(pos_counts) * 1.5:
        print("⚖️ Applying SMOTE to POS data...")
        original_shape = X_p.shape
        X_p_flat = X_p.reshape(X_p.shape[0], -1)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_p_flat, y_p_balanced = smote.fit_resample(X_p_flat, y_p)
        X_p_balanced = X_p_flat.reshape(-1, *original_shape[1:])
    
    print(f"📈 Final data shapes:")
    print(f"Emotion: {X_e.shape} -> {X_e_improved.shape}")
    print(f"POS: {X_p.shape} -> {X_p_balanced.shape}")
    
    # Add channel dimension
    X_e_improved = X_e_improved[..., np.newaxis]
    X_p_balanced = X_p_balanced[..., np.newaxis]
    channel_weights = np.array([1.0, 1.0, 0.75, 0.75]).reshape(1, 4, 1, 1)
    # Combine datasets
    X_e_improved = X_e_improved * channel_weights
    
    X_combined = np.concatenate([X_e_improved, X_p_balanced], axis=0)
    
    y_emotion_combined = np.concatenate([
        to_categorical(y_e_improved, NUM_CLASSES_EMOTION),
        np.zeros((len(X_p_balanced), NUM_CLASSES_EMOTION))
    ], axis=0)
    
    y_pos_combined = np.concatenate([
        np.zeros((len(X_e_improved), NUM_CLASSES_POS)),
        to_categorical(y_p_balanced, NUM_CLASSES_POS)
    ], axis=0)
    
    # Sample weights
    sw_emotion = np.concatenate([np.ones(len(X_e_improved)), np.zeros(len(X_p_balanced))], axis=0)
    sw_pos = np.concatenate([np.zeros(len(X_e_improved)), np.ones(len(X_p_balanced))], axis=0)
    
    # Stratified splitting
    print("🎯 Creating stratified train/validation split...")
    train_idx, val_idx = create_stratified_split(
        X_combined, y_e_improved, y_p_balanced, test_size=0.2, random_state=RANDOM_STATE
    )
    
    X_train, X_val = X_combined[train_idx], X_combined[val_idx]
    ye_train, ye_val = y_emotion_combined[train_idx], y_emotion_combined[val_idx]
    yp_train, yp_val = y_pos_combined[train_idx], y_pos_combined[val_idx]
    swe_train, swe_val = sw_emotion[train_idx], sw_emotion[val_idx]
    swp_train, swp_val = sw_pos[train_idx], sw_pos[val_idx]
    
    # -------------------------
    # BALANCED COMPILATION AND TRAINING
    # -------------------------
    print("⚙️ Compiling model with balanced approach...")
    # combined_model.compile(
    #     optimizer=Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
    #     loss=[
    #         balanced_focal_loss(alpha=0.25, gamma=1.5),  # Gentle focal loss
    #         balanced_focal_loss(alpha=0.25, gamma=1.5)   # Same for both tasks
    #     ],
    #     metrics=[['accuracy'], ['accuracy']],
    #     loss_weights=[1.2, 1.0]  # Only slight emphasis on emotion
    # )
    
    # Standard callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='actually_weighted_best_balanced_fusion_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("🏋️ Starting balanced training...")
    # history = combined_model.fit(
    #     X_train,
    #     [ye_train, yp_train],
    #     sample_weight=[swe_train, swp_train],
    #     validation_data=(X_val, [ye_val, yp_val], [swe_val, swp_val]),
    #     epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     callbacks=callbacks,
    #     verbose=1
    # )
    
    # -------------------------
    # EVALUATION
    # -------------------------
    print("📊 Evaluation...")
    
    # Load best model
    best_model = load_model(
        'actually_weighted_best_balanced_fusion_model.h5', 
        custom_objects={
            'focal_loss_fixed': balanced_focal_loss()
        }
    )
    
    # Predictions
    y_pred_emotion, y_pred_pos = best_model.predict(X_val)
    y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
    y_pred_pos_classes = np.argmax(y_pred_pos, axis=1)
    
    ye_val_labels = np.argmax(ye_val, axis=1)
    yp_val_labels = np.argmax(yp_val, axis=1)
    
    # Emotion task evaluation
    emotion_mask = swe_val > 0
    if np.sum(emotion_mask) > 0:
        print("\n🎭 BALANCED EMOTION CLASSIFICATION RESULTS:")
        print(classification_report(
            ye_val_labels[emotion_mask], 
            y_pred_emotion_classes[emotion_mask],
            target_names=["Negative", "Neutral", "Positive"]
        ))
        
        emotion_cm = confusion_matrix(ye_val_labels[emotion_mask], y_pred_emotion_classes[emotion_mask])
        print("Emotion Confusion Matrix:")
        print(emotion_cm)
        
        # Calculate per-class accuracy
        emotion_per_class_acc = np.diag(emotion_cm) / np.sum(emotion_cm, axis=1) * 100
        print(f"Per-class accuracy: Negative: {emotion_per_class_acc[0]:.1f}%, Neutral: {emotion_per_class_acc[1]:.1f}%, Positive: {emotion_per_class_acc[2]:.1f}%")
    
    # POS task evaluation
    pos_mask = swp_val > 0
    if np.sum(pos_mask) > 0:
        print("\n📝 POS CLASSIFICATION RESULTS:")
        print(classification_report(
            yp_val_labels[pos_mask], 
            y_pred_pos_classes[pos_mask],
            target_names=["Noun", "Verb"]
        ))
    
    print("\n✅ Balanced multi-task fusion training completed!")
    print(f"📁 Best model saved as: best_balanced_fusion_model.h5")
    
    return best_model, history

if __name__ == "__main__":
    model, training_history = main()



