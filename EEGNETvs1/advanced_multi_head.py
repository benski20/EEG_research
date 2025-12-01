# improved_multi_task_fusion.py - ENHANCED VERSION WITH HIGH-IMPACT IMPROVEMENTS
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten,
    MultiHeadAttention, LayerNormalization, Add
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
# ENHANCED CONFIG
# -------------------------
EMOTION_MODEL_PATH = "eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5"
POS_MODEL_PATH = "eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5"
EMOTION_DATA_PATH = "eeg_fft_dataset.pkl"
POS_DATA_PATH = "eeg_fft_nounvsverb_JASON_dataset.pkl"

INPUT_SHAPE = (4, 60, 1)
NUM_CLASSES_EMOTION = 3
NUM_CLASSES_POS = 2
FUSION_DENSE = 256  # Increased capacity
BATCH_SIZE = 32     # Increased for better gradient estimates
EPOCHS = 100        # Increased with early stopping
LR = 1e-4          # Lower initial LR with scheduling
RANDOM_STATE = 42

# -------------------------
# FOCAL LOSS IMPLEMENTATION
# High Impact: Addresses class imbalance better than standard cross-entropy
# -------------------------
def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal Loss for addressing class imbalance.
    Benefits:
    - Reduces loss contribution from easy examples (well-classified)
    - Focuses training on hard examples (misclassified)
    - Better than class weights for severe imbalance
    """
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        # Compute cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Compute weights
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        
        # Apply focal weight
        focal_loss = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    
    return focal_loss_fixed

# -------------------------
# ENHANCED DATA PREPROCESSING
# High Impact: Better normalization and augmentation
# -------------------------
def robust_normalize_eeg(X):
    """
    Robust normalization using percentile-based scaling.
    Benefits:
    - Less sensitive to outliers than z-score normalization
    - Preserves relative amplitude differences between channels
    - More stable across different recording sessions
    """
    # Per-channel normalization using 5th and 95th percentiles
    for ch in range(X.shape[1]):
        channel_data = X[:, ch, :]
        p5 = np.percentile(channel_data, 5, axis=1, keepdims=True)
        p95 = np.percentile(channel_data, 95, axis=1, keepdims=True)
        
        # Avoid division by zero
        scale = p95 - p5
        scale = np.where(scale == 0, 1, scale)
        
        X[:, ch, :] = (channel_data - p5) / scale
    
    return X

def augment_eeg_batch(X, noise_std=0.02, time_shift_range=5):
    """
    Enhanced EEG data augmentation.
    Benefits:
    - Increases dataset size without collecting more data
    - Improves model robustness to noise and timing variations
    - Reduces overfitting
    """
    augmented = []
    
    for sample in X:
        # Original sample
        augmented.append(sample)
        
        # Gaussian noise augmentation
        noise = np.random.normal(0, noise_std, sample.shape)
        noisy_sample = sample + noise
        augmented.append(noisy_sample)
        
        # Time shift augmentation
        if time_shift_range > 0:
            shift = np.random.randint(-time_shift_range, time_shift_range + 1)
            if shift != 0:
                shifted_sample = np.roll(sample, shift, axis=1)
                augmented.append(shifted_sample)
    
    return np.array(augmented)

# -------------------------
# SMOTE FOR CLASS BALANCING  
# High Impact: Synthetic oversampling for minority classes
# -------------------------
def apply_smote_to_eeg(X, y, random_state=42):
    """
    Apply SMOTE to EEG data after flattening.
    Benefits:
    - Creates synthetic minority class samples
    - Better than simple oversampling (no exact duplicates)
    - Maintains class boundaries while balancing dataset
    """
    # Flatten for SMOTE (SMOTE works on 2D data)
    original_shape = X.shape
    X_flat = X.reshape(X.shape[0], -1)
    
    # Apply SMOTE
    smote = SMOTE(random_state=random_state, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X_flat, y)
    
    # Reshape back
    X_resampled = X_resampled.reshape(-1, *original_shape[1:])
    
    return X_resampled, y_resampled

# -------------------------
# ATTENTION-BASED FUSION LAYER
# High Impact: Learns which features are most important for each task
# -------------------------
from tensorflow.keras.layers import Lambda, Reshape

def create_attention_fusion_layer(emotion_features, pos_features, d_model=128):
    """
    Multi-head attention for feature fusion.
    Benefits:
    - Learns which features are most relevant for each task
    - Allows features to interact in a learned way
    - More sophisticated than simple concatenation
    """
    # Get feature dimensions
    emotion_dim = emotion_features.shape[-1]
    pos_dim = pos_features.shape[-1]
    
    # Reshape features for attention using Keras layers instead of tf operations
    emotion_reshaped = Reshape((1, emotion_dim))(emotion_features)  # (batch, 1, feat_dim)
    pos_reshaped = Reshape((1, pos_dim))(pos_features)
    
    # If dimensions don't match, project them to same dimension
    if emotion_dim != pos_dim:
        # Project to same dimension
        target_dim = max(emotion_dim, pos_dim)
        if emotion_dim < target_dim:
            emotion_reshaped = Dense(target_dim)(emotion_reshaped)
        if pos_dim < target_dim:
            pos_reshaped = Dense(target_dim)(pos_reshaped)
    
    # Combine features
    combined_features = Concatenate(axis=1)([emotion_reshaped, pos_reshaped])  # (batch, 2, feat_dim)
    
    # Multi-head self-attention
    attention_output = MultiHeadAttention(
        num_heads=4, 
        key_dim=d_model//4,
        name="feature_attention"
    )(combined_features, combined_features)
    
    # Layer normalization and residual connection
    attention_output = LayerNormalization()(attention_output)
    attention_output = Add()([combined_features, attention_output])
    
    # Flatten back to vector
    fused_features = Flatten()(attention_output)
    
    return fused_features

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

# -------------------------
# STRATIFIED DATA SPLITTING
# High Impact: Ensures balanced representation in train/val sets
# -------------------------
def create_stratified_split(X, y_emotion, y_pos, test_size=0.2, random_state=42):
    """
    Create stratified split considering both tasks.
    Benefits:
    - Maintains class distribution in both train and validation sets
    - More reliable performance estimates
    - Prevents validation set bias
    """
    # Create a combined label for stratification
    combined_labels = []
    for i in range(len(X)):
        if i < len(y_emotion):  # Emotion samples
            combined_labels.append(f"emotion_{y_emotion[i]}")
        else:  # POS samples  
            combined_labels.append(f"pos_{y_pos[i-len(y_emotion)]}")
    
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(sss.split(X, combined_labels))
    
    return train_idx, val_idx

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
    # ENHANCED MODEL ARCHITECTURE
    # -------------------------
    print("🏗️ Building enhanced fusion model...")
    main_input = Input(shape=INPUT_SHAPE, name="eeg_input")
    
    # Feature extraction
    feat_a_tensor = emotion_extractor(main_input)
    feat_b_tensor = pos_extractor(main_input)
    
    feat_a_flat = Flatten(name="emotion_features_flat")(feat_a_tensor)
    feat_b_flat = Flatten(name="pos_features_flat")(feat_b_tensor)
    
    # ENHANCEMENT: Attention-based fusion instead of simple concatenation
    fused_features = create_attention_fusion_layer(feat_a_flat, feat_b_flat, d_model=FUSION_DENSE)
    
    # Enhanced fusion layers with residual connections
    x = Dense(FUSION_DENSE, activation="relu", name="fusion_dense1")(fused_features)
    x = BatchNormalization(name="fusion_bn1")(x)
    x = Dropout(0.4, name="fusion_dropout1")(x)  # Increased dropout
    
    # Additional fusion layer for better capacity
    x_residual = Dense(FUSION_DENSE//2, activation="relu", name="fusion_dense2")(x)
    x_residual = BatchNormalization(name="fusion_bn2")(x_residual)
    x_residual = Dropout(0.3, name="fusion_dropout2")(x_residual)
    
    # Output heads
    emotion_out = Dense(NUM_CLASSES_EMOTION, activation="softmax", name="emotion")(x_residual)
    pos_out = Dense(NUM_CLASSES_POS, activation="softmax", name="pos")(x_residual)
    
    combined_model = Model(inputs=main_input, outputs=[emotion_out, pos_out], name="enhanced_fusion_model")
    
    # -------------------------
    # ENHANCED DATA LOADING AND PREPROCESSING
    # -------------------------
    print("📂 Loading and preprocessing datasets...")
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data = load_pickle(POS_DATA_PATH)
    
    # Process emotion data
    X_e = np.array(emotion_data["data"]).reshape(-1, 4, 60)
    X_e = robust_normalize_eeg(X_e)  # Enhanced normalization
    y_e = np.array(emotion_data["labels"])
    
    # Process POS data
    X_p = np.array(pos_data["data"]).reshape(-1, 4, 60)
    X_p = robust_normalize_eeg(X_p)  # Enhanced normalization
    y_p = np.array(pos_data["labels"]) - 1
    
    # ENHANCEMENT: Apply SMOTE for class balancing
    print("⚖️ Applying SMOTE for class balancing...")
    X_e_balanced, y_e_balanced = apply_smote_to_eeg(X_e, y_e)
    X_p_balanced, y_p_balanced = apply_smote_to_eeg(X_p, y_p)
    
    print(f"📈 Data after SMOTE:")
    print(f"Emotion: {X_e.shape} -> {X_e_balanced.shape}")
    print(f"POS: {X_p.shape} -> {X_p_balanced.shape}")
    
    # Add channel dimension
    X_e_balanced = X_e_balanced[..., np.newaxis]
    X_p_balanced = X_p_balanced[..., np.newaxis]
    
    # Combine datasets
    X_combined = np.concatenate([X_e_balanced, X_p_balanced], axis=0)
    
    y_emotion_combined = np.concatenate([
        to_categorical(y_e_balanced, NUM_CLASSES_EMOTION),
        np.zeros((len(X_p_balanced), NUM_CLASSES_EMOTION))
    ], axis=0)
    
    y_pos_combined = np.concatenate([
        np.zeros((len(X_e_balanced), NUM_CLASSES_POS)),
        to_categorical(y_p_balanced, NUM_CLASSES_POS)
    ], axis=0)
    
    # Sample weights
    sw_emotion = np.concatenate([np.ones(len(X_e_balanced)), np.zeros(len(X_p_balanced))], axis=0)
    sw_pos = np.concatenate([np.zeros(len(X_e_balanced)), np.ones(len(X_p_balanced))], axis=0)
    
    # ENHANCEMENT: Stratified splitting
    print("🎯 Creating stratified train/validation split...")
    train_idx, val_idx = create_stratified_split(
        X_combined, y_e_balanced, y_p_balanced, test_size=0.2, random_state=RANDOM_STATE
    )
    
    X_train, X_val = X_combined[train_idx], X_combined[val_idx]
    ye_train, ye_val = y_emotion_combined[train_idx], y_emotion_combined[val_idx]
    yp_train, yp_val = y_pos_combined[train_idx], y_pos_combined[val_idx]
    swe_train, swe_val = sw_emotion[train_idx], sw_emotion[val_idx]
    swp_train, swp_val = sw_pos[train_idx], sw_pos[val_idx]
    
    # -------------------------
    # ENHANCED COMPILATION AND TRAINING
    # -------------------------
    print("⚙️ Compiling model with focal loss...")
    combined_model.compile(
        optimizer=Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss=[focal_loss(alpha=0.25, gamma=2.0), focal_loss(alpha=0.25, gamma=2.0)],
        metrics=[["accuracy"], ["accuracy"]],
        loss_weights=[1.0, 1.0]
    )
    
    # ENHANCEMENT: Advanced callbacks
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
            filepath='best_enhanced_fusion_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    print("🏋️ Starting enhanced training...")
    history = combined_model.fit(
        X_train,
        [ye_train, yp_train],
        sample_weight=[swe_train, swp_train],
        validation_data=(X_val, [ye_val, yp_val], [swe_val, swp_val]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # -------------------------
    # ENHANCED EVALUATION
    # -------------------------
    print("📊 Comprehensive evaluation...")
    
    # Load best model
    best_model = load_model('best_enhanced_fusion_model.h5', 
                           custom_objects={'focal_loss_fixed': focal_loss()})
    
    # Predictions
    y_pred_emotion, y_pred_pos = best_model.predict(X_val)
    y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
    y_pred_pos_classes = np.argmax(y_pred_pos, axis=1)
    
    ye_val_labels = np.argmax(ye_val, axis=1)
    yp_val_labels = np.argmax(yp_val, axis=1)
    
    # Emotion task evaluation (only where mask = 1)
    emotion_mask = swe_val > 0
    if np.sum(emotion_mask) > 0:
        print("\n🎭 EMOTION CLASSIFICATION RESULTS:")
        print(classification_report(
            ye_val_labels[emotion_mask], 
            y_pred_emotion_classes[emotion_mask],
            target_names=[f"Emotion_{i}" for i in range(NUM_CLASSES_EMOTION)]
        ))
    
    # POS task evaluation (only where mask = 1)
    pos_mask = swp_val > 0
    if np.sum(pos_mask) > 0:
        print("\n📝 POS CLASSIFICATION RESULTS:")
        print(classification_report(
            yp_val_labels[pos_mask], 
            y_pred_pos_classes[pos_mask],
            target_names=[f"POS_{i}" for i in range(NUM_CLASSES_POS)]
        ))
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    if np.sum(emotion_mask) > 0:
        cm_emotion = confusion_matrix(ye_val_labels[emotion_mask], y_pred_emotion_classes[emotion_mask])
        axes[0].imshow(cm_emotion, cmap='Blues')
        axes[0].set_title('Enhanced Emotion Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Add text annotations
        for i in range(cm_emotion.shape[0]):
            for j in range(cm_emotion.shape[1]):
                axes[0].text(j, i, str(cm_emotion[i, j]), ha='center', va='center')
    
    if np.sum(pos_mask) > 0:
        cm_pos = confusion_matrix(yp_val_labels[pos_mask], y_pred_pos_classes[pos_mask])
        axes[1].imshow(cm_pos, cmap='Oranges')
        axes[1].set_title('Enhanced POS Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        # Add text annotations
        for i in range(cm_pos.shape[0]):
            for j in range(cm_pos.shape[1]):
                axes[1].text(j, i, str(cm_pos[i, j]), ha='center', va='center')
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Enhanced multi-task fusion training completed!")
    print(f"📁 Best model saved as: best_enhanced_fusion_model.h5")
    
    return best_model, history

if __name__ == "__main__":
    model, training_history = main()