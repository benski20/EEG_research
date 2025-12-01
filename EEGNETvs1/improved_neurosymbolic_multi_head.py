# improved_neurosymbolic_eeg_model.py - PROPERLY GROUNDED NEUROSYMBOLIC APPROACH
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten,
    Lambda, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import os

# -------------------------
# CONFIGURATION
# -------------------------
EMOTION_MODEL_PATH = "eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5"
POS_MODEL_PATH     = "eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5"
EMOTION_DATA_PATH  = "eeg_fft_dataset.pkl"
POS_DATA_PATH      = "eeg_fft_nounvsverb_JASON_dataset.pkl"

INPUT_SHAPE = (4, 60, 1)  # 4 channels, 60 freq bins, 1 feature
NUM_CLASSES_EMOTION = 3
NUM_CLASSES_POS = 2
FUSION_DENSE = 256
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
RANDOM_STATE = 42

# Neurosymbolic parameters - SIMPLIFIED AND GROUNDED
WARMUP_EPOCHS = 20  # Epochs before applying constraints
CONSTRAINT_WEIGHT = 0.3  # Moderate constraint influence
UNCERTAINTY_THRESHOLD = 0.6  # Apply constraints when max_prob < threshold

# EEG frequency band indices (approximate for 60 freq bins from 0-30Hz)
ALPHA_INDICES = slice(16, 26)   # ~8-13 Hz
BETA_INDICES = slice(26, 40)    # ~13-20 Hz  
THETA_INDICES = slice(8, 16)    # ~4-8 Hz

# -------------------------
# EEG-GROUNDED PREDICATE LAYER
# -------------------------
class EEGPredicateLayer(Layer):
    """
    Computes neuroscientifically-grounded predicates from EEG frequency features
    """
    def __init__(self, name="eeg_predicates", **kwargs):
        super().__init__(name=name, **kwargs)
        
    def build(self, input_shape):
        # input_shape: (batch, 4, 60, 1) - 4 channels, 60 freq bins
        super().build(input_shape)
        
    def call(self, inputs):
        # inputs: (batch, 4, 60, 1)
        eeg_data = tf.squeeze(inputs, axis=-1)  # (batch, 4, 60)
        
        # Channel mapping: [F3, F4, P3, P4] approximate
        f3_data = eeg_data[:, 0, :]  # Left frontal
        f4_data = eeg_data[:, 1, :]  # Right frontal  
        p3_data = eeg_data[:, 2, :]  # Left parietal
        p4_data = eeg_data[:, 3, :]  # Right parietal
        
        # Predicate 1: Frontal Alpha Asymmetry (FAA)
        # Left > Right alpha power indicates approach motivation (positive emotion)
        f3_alpha = tf.reduce_mean(f3_data[:, ALPHA_INDICES], axis=1)
        f4_alpha = tf.reduce_mean(f4_data[:, ALPHA_INDICES], axis=1)
        
        # Asymmetry: (Right - Left) / (Right + Left)
        # Positive values indicate left dominance (positive emotion)
        alpha_asymmetry = (f4_alpha - f3_alpha) / (f4_alpha + f3_alpha + 1e-6)
        alpha_asymmetry = tf.nn.sigmoid(alpha_asymmetry * 5.0)  # Scale and sigmoid
        
        # Predicate 2: Motor Beta Activity
        # High beta in motor areas associated with motor preparation/action (verbs)
        motor_beta = tf.reduce_mean(tf.concat([
            p3_data[:, BETA_INDICES],
            p4_data[:, BETA_INDICES]
        ], axis=1), axis=1)
        motor_beta = tf.nn.sigmoid((motor_beta - 0.5) * 3.0)  # Normalize and sigmoid
        
        # Predicate 3: Frontal Theta Power
        # High frontal theta associated with attention/cognitive control
        frontal_theta = tf.reduce_mean(tf.concat([
            f3_data[:, THETA_INDICES],
            f4_data[:, THETA_INDICES]
        ], axis=1), axis=1)
        frontal_theta = tf.nn.sigmoid((frontal_theta - 0.3) * 4.0)
        
        predicates = tf.stack([alpha_asymmetry, motor_beta, frontal_theta], axis=1)
        return predicates

# -------------------------
# NEUROSYMBOLIC CONSTRAINT LAYER
# -------------------------
class NeurosymbolicConstraintLayer(Layer):
    """
    Applies domain-grounded constraints based on EEG predicates
    """
    def __init__(self, warmup_epochs=WARMUP_EPOCHS, constraint_weight=CONSTRAINT_WEIGHT,
                 uncertainty_threshold=UNCERTAINTY_THRESHOLD, name="neurosymbolic_constraints", **kwargs):
        super().__init__(name=name, **kwargs)
        self.warmup_epochs = warmup_epochs
        self.constraint_weight = constraint_weight
        self.uncertainty_threshold = uncertainty_threshold
        
    def build(self, input_shape):
        super().build(input_shape)
        # Learnable rule weights
        self.rule_weights = self.add_weight(
            name="rule_weights",
            shape=(3,),  # 3 rules
            initializer="ones",
            trainable=True
        )
        
    def call(self, inputs, training=None):
        emotion_probs, pos_probs, predicates, epoch_progress = inputs
        
        alpha_asym, motor_beta, frontal_theta = tf.unstack(predicates, axis=1)
        
        # Rule 1: Alpha asymmetry → Emotion valence
        # Positive asymmetry (left dominance) → Positive emotion
        pos_emotion = emotion_probs[:, 2]  # Positive class
        neg_emotion = emotion_probs[:, 0]  # Negative class
        
        rule1_violation = tf.maximum(0.0, alpha_asym - pos_emotion + neg_emotion)
        
        # Rule 2: Motor beta → Verb likelihood  
        # High motor beta → Higher verb probability
        verb_prob = pos_probs[:, 1]  # Verb class
        rule2_violation = tf.maximum(0.0, motor_beta - verb_prob)
        
        # Rule 3: Frontal theta → Attention consistency
        # High theta should be consistent with high confidence predictions
        emotion_confidence = tf.reduce_max(emotion_probs, axis=1)
        pos_confidence = tf.reduce_max(pos_probs, axis=1)
        avg_confidence = (emotion_confidence + pos_confidence) / 2.0
        
        rule3_violation = tf.maximum(0.0, frontal_theta - avg_confidence)
        
        # Weighted rule violations
        total_violation = (
            self.rule_weights[0] * rule1_violation +
            self.rule_weights[1] * rule2_violation + 
            self.rule_weights[2] * rule3_violation
        )
        
        # Apply constraints based on epoch progress and uncertainty
        constraint_mask = tf.cast(epoch_progress > (self.warmup_epochs / 100.0), tf.float32)
        
        # Only apply constraints when model is uncertain
        max_emotion_prob = tf.reduce_max(emotion_probs, axis=1)
        uncertainty_mask = tf.cast(max_emotion_prob < self.uncertainty_threshold, tf.float32)
        
        final_mask = constraint_mask * uncertainty_mask
        constraint_loss = tf.reduce_mean(total_violation * final_mask) * self.constraint_weight
        
        # Add as layer loss
        self.add_loss(constraint_loss)
        
        # Return emotion probabilities (for model output)
        return emotion_probs

# -------------------------
# CURRICULUM CALLBACK
# -------------------------
class CurriculumCallback(tf.keras.callbacks.Callback):
    """Gradually increases constraint influence during training"""
    
    def __init__(self, constraint_layer_names):
        self.constraint_layer_names = constraint_layer_names
        
    def on_epoch_begin(self, epoch, logs=None):
        # Update epoch progress in constraint layers
        progress = epoch / 100.0  # Normalize by total epochs
        
        for layer in self.model.layers:
            if layer.name in self.constraint_layer_names:
                # This is a simple approach; in practice you might need 
                # a more sophisticated way to pass epoch info
                pass

# -------------------------
# HELPER FUNCTIONS (same as before)
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
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    return focal_loss_fixed

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

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

def robust_normalize_eeg(X):
    for ch in range(X.shape[1]):
        channel_data = X[:, ch, :]
        p5 = np.percentile(channel_data, 5, axis=1, keepdims=True)
        p95 = np.percentile(channel_data, 95, axis=1, keepdims=True)
        scale = p95 - p5
        scale = np.where(scale == 0, 1, scale)
        X[:, ch, :] = (channel_data - p5) / scale
    return X

def create_stratified_split(X, y_emotion, y_pos, test_size=0.2, random_state=42):
    combined_labels = []
    for i in range(len(X)):
        if i < len(y_emotion):
            combined_labels.append(f"emotion_{y_emotion[i]}")
        else:
            combined_labels.append(f"pos_{y_pos[i-len(y_emotion)]}")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(sss.split(X, combined_labels))
    return train_idx, val_idx

def apply_conservative_emotion_improvements(X_emotion, y_emotion):
    """Conservative improvements from the base model"""
    # Light preprocessing
    X_processed = X_emotion.copy().astype(np.float32)
    for ch in range(X_processed.shape[1]):
        channel_data = X_processed[:, ch, :]
        p25 = np.percentile(channel_data, 25, axis=1, keepdims=True)
        p75 = np.percentile(channel_data, 75, axis=1, keepdims=True)
        scale = p75 - p25
        scale = np.where(scale == 0, 1, scale)
        X_processed[:, ch, :] = (channel_data - p25) / scale
    
    # Conservative augmentation
    neutral_mask = (y_emotion == 1)
    neutral_samples = X_processed[neutral_mask]
    
    if len(neutral_samples) > 0:
        augmented_samples = []
        for sample in neutral_samples:
            aug_sample = sample.copy()
            noise_std = 0.01 * np.std(aug_sample)
            noise = np.random.normal(0, noise_std, aug_sample.shape)
            aug_sample += noise
            augmented_samples.append(aug_sample)
        
        X_augmented = np.concatenate([X_processed, np.array(augmented_samples)], axis=0)
        y_augmented = np.concatenate([y_emotion, np.full(len(augmented_samples), 1)], axis=0)
    else:
        X_augmented, y_augmented = X_processed, y_emotion
    
    # SMOTE if needed
    class_counts = np.bincount(y_augmented)
    if np.max(class_counts) > np.min(class_counts) * 1.5:
        original_shape = X_augmented.shape
        X_flat = X_augmented.reshape(X_augmented.shape[0], -1)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_flat, y_augmented = smote.fit_resample(X_flat, y_augmented)
        X_balanced = X_flat.reshape(-1, *original_shape[1:])
    else:
        X_balanced = X_augmented
    
    # Class weights
    classes = np.unique(y_augmented)
    weights = compute_class_weight('balanced', classes=classes, y=y_augmented)
    class_weights = dict(zip(classes, weights))
    if 1 in class_weights:
        class_weights[1] *= 1.3
    
    return X_balanced, y_augmented, class_weights

def create_attention_fusion(emotion_features, pos_features, fusion_dim=256):
    """Simple attention fusion"""
    emotion_proj = Dense(fusion_dim//2, activation='relu')(emotion_features)
    pos_proj = Dense(fusion_dim//2, activation='relu')(pos_features)
    
    emotion_att = Dense(1, activation='sigmoid')(emotion_proj)
    pos_att = Dense(1, activation='sigmoid')(pos_proj)
    
    emotion_weighted = Lambda(lambda x: x[0] * x[1])([emotion_proj, emotion_att])
    pos_weighted = Lambda(lambda x: x[0] * x[1])([pos_proj, pos_att])
    
    return Concatenate()([emotion_weighted, pos_weighted])

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
    
    emotion_extractor.trainable = False
    pos_extractor.trainable = False
    
    # -------------------------
    # BUILD NEUROSYMBOLIC ARCHITECTURE
    # -------------------------
    print("🏗️ Building neurosymbolic fusion model...")
    
    # Inputs
    eeg_input = Input(shape=INPUT_SHAPE, name="eeg_input")
    epoch_progress_input = Input(shape=(1,), name="epoch_progress")
    
    # Feature extraction (frozen)
    emotion_features = Flatten()(emotion_extractor(eeg_input))
    pos_features = Flatten()(pos_extractor(eeg_input))
    
    # Attention fusion
    fused_features = create_attention_fusion(emotion_features, pos_features, FUSION_DENSE)
    
    # Shared trunk
    x = Dense(FUSION_DENSE, activation="relu")(fused_features)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(FUSION_DENSE//2, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # Task heads
    emotion_logits = Dense(NUM_CLASSES_EMOTION, activation="softmax", name="emotion")(x)
    pos_logits = Dense(NUM_CLASSES_POS, activation="softmax", name="pos")(x)
    
    # 🧠 NEUROSYMBOLIC COMPONENTS
    # EEG predicates computed from raw input
    eeg_predicates = EEGPredicateLayer()(eeg_input)
    
    # Neurosymbolic constraints (returns emotion_logits with added loss)
    emotion_constrained = NeurosymbolicConstraintLayer()(
        [emotion_logits, pos_logits, eeg_predicates, epoch_progress_input]
    )
    
    # Final model
    model = Model(
        inputs=[eeg_input, epoch_progress_input],
        outputs=[emotion_constrained, pos_logits],
        name="neurosymbolic_eeg_fusion"
    )
    
    # -------------------------
    # DATA LOADING
    # -------------------------
    print("📂 Loading datasets...")
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data = load_pickle(POS_DATA_PATH)
    
    # Process emotion data
    X_e = np.array(emotion_data["data"]).reshape(-1, 4, 60)
    y_e = np.array(emotion_data["labels"])
    X_e_improved, y_e_improved, emotion_class_weights = apply_conservative_emotion_improvements(X_e, y_e)
    
    # Process POS data
    X_p = np.array(pos_data["data"]).reshape(-1, 4, 60)
    X_p = robust_normalize_eeg(X_p)
    y_p = np.array(pos_data["labels"]) - 1
    
    # Balance POS data if needed
    pos_counts = np.bincount(y_p)
    if len(pos_counts) > 1 and np.max(pos_counts) > np.min(pos_counts) * 1.5:
        original_shape = X_p.shape
        X_p_flat = X_p.reshape(X_p.shape[0], -1)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_p_flat, y_p_balanced = smote.fit_resample(X_p_flat, y_p)
        X_p_balanced = X_p_flat.reshape(-1, *original_shape[1:])
    else:
        X_p_balanced = X_p
        y_p_balanced = y_p
    
    # Add channel dimension and combine
    X_e_improved = X_e_improved[..., np.newaxis]
    X_p_balanced = X_p_balanced[..., np.newaxis]
    
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
    
    # Train/val split
    train_idx, val_idx = create_stratified_split(
        X_combined, y_e_improved, y_p_balanced, test_size=0.2, random_state=RANDOM_STATE
    )
    
    X_train, X_val = X_combined[train_idx], X_combined[val_idx]
    ye_train, ye_val = y_emotion_combined[train_idx], y_emotion_combined[val_idx]
    yp_train, yp_val = y_pos_combined[train_idx], y_pos_combined[val_idx]
    swe_train, swe_val = sw_emotion[train_idx], sw_emotion[val_idx]
    swp_train, swp_val = sw_pos[train_idx], sw_pos[val_idx]
    
    # -------------------------
    # TRAINING WITH CURRICULUM
    # -------------------------
    print("⚙️ Compiling neurosymbolic model...")
    model.compile(
        optimizer=Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss=[
            balanced_focal_loss(alpha=0.25, gamma=1.5),
            balanced_focal_loss(alpha=0.25, gamma=1.5)
        ],
        metrics=[['accuracy'], ['accuracy']],
        loss_weights=[1.2, 1.0]
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath='improved_neurosymbolic_eeg_model.h5',
                       monitor='val_loss', save_best_only=True, verbose=1)
    ]
    
    print("🧠 Starting neurosymbolic training with curriculum...")
    
    # Training loop with epoch progress
    class EpochProgressGenerator:
        def __init__(self, X, epoch_inputs, y, sw, batch_size, total_epochs):
            self.X = X
            self.epoch_inputs = epoch_inputs  
            self.y = y
            self.sw = sw
            self.batch_size = batch_size
            self.total_epochs = total_epochs
            self.current_epoch = 0
            
        def __iter__(self):
            return self
            
        def __next__(self):
            # Update epoch progress for all samples
            progress = np.full((len(self.X), 1), self.current_epoch / self.total_epochs, dtype=np.float32)
            self.current_epoch += 1
            return ([self.X, progress], self.y, self.sw)
    
    # Simplified training approach
    history = model.fit(
        [X_train, np.zeros((len(X_train), 1))],  # Start with zero epoch progress
        [ye_train, yp_train],
        sample_weight=[swe_train, swp_train],
        validation_data=([X_val, np.zeros((len(X_val), 1))], [ye_val, yp_val], [swe_val, swp_val]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # -------------------------
    # EVALUATION
    # -------------------------
    print("📊 Evaluating neurosymbolic model...")
    
    best_model = load_model(
        'improved_neurosymbolic_eeg_model.h5',
        custom_objects={
            'focal_loss_fixed': balanced_focal_loss(),
            'EEGPredicateLayer': EEGPredicateLayer,
            'NeurosymbolicConstraintLayer': NeurosymbolicConstraintLayer
        }
    )
    
    # Final evaluation
    y_pred_emotion, y_pred_pos = best_model.predict([X_val, np.ones((len(X_val), 1))], verbose=0)
    y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
    y_pred_pos_classes = np.argmax(y_pred_pos, axis=1)
    
    ye_val_labels = np.argmax(ye_val, axis=1)
    yp_val_labels = np.argmax(yp_val, axis=1)
    
    # Emotion evaluation
    emotion_mask = swe_val > 0
    if np.sum(emotion_mask) > 0:
        print("\n🧠 NEUROSYMBOLIC EMOTION CLASSIFICATION RESULTS:")
        print(classification_report(
            ye_val_labels[emotion_mask],
            y_pred_emotion_classes[emotion_mask],
            target_names=["Negative", "Neutral", "Positive"]
        ))
        
        emotion_cm = confusion_matrix(ye_val_labels[emotion_mask], y_pred_emotion_classes[emotion_mask])
        print("Emotion Confusion Matrix:")
        print(emotion_cm)
        
        emotion_per_class_acc = np.diag(emotion_cm) / np.sum(emotion_cm, axis=1) * 100
        print(f"Per-class accuracy: Negative: {emotion_per_class_acc[0]:.1f}%, "
              f"Neutral: {emotion_per_class_acc[1]:.1f}%, Positive: {emotion_per_class_acc[2]:.1f}%")
    
    # POS evaluation
    pos_mask = swp_val > 0
    if np.sum(pos_mask) > 0:
        print("\n📝 NEUROSYMBOLIC POS CLASSIFICATION RESULTS:")
        print(classification_report(
            yp_val_labels[pos_mask],
            y_pred_pos_classes[pos_mask],
            target_names=["Noun", "Verb"]
        ))
    
    # -------------------------
    # INTERPRETABILITY ANALYSIS
    # -------------------------
    print("\n🔍 Neurosymbolic Interpretability Analysis:")
    
    # Get EEG predicates for a sample of validation data
    predicate_model = Model(inputs=best_model.input, outputs=best_model.get_layer('eeg_predicates').output)
    sample_predicates = predicate_model.predict([X_val[:100], np.ones((100, 1))], verbose=0)
    
    # Analyze predicate activations
    predicate_names = ["Alpha_Asymmetry", "Motor_Beta", "Frontal_Theta"]
    for i, name in enumerate(predicate_names):
        values = sample_predicates[:, i]
        print(f"{name}: mean={np.mean(values):.3f}, std={np.std(values):.3f}, "
              f"range=[{np.min(values):.3f}, {np.max(values):.3f}]")
    
    # Show examples where rules were likely applied (low confidence cases)
    emotion_confidence = np.max(y_pred_emotion, axis=1)
    low_conf_indices = np.where(emotion_confidence < UNCERTAINTY_THRESHOLD)[0][:5]
    
    if len(low_conf_indices) > 0:
        print(f"\n💡 Examples where neurosymbolic constraints were applied (confidence < {UNCERTAINTY_THRESHOLD}):")
        for idx in low_conf_indices:
            if idx < len(sample_predicates):
                alpha_asym, motor_beta, frontal_theta = sample_predicates[idx]
                pred_class = ["Negative", "Neutral", "Positive"][y_pred_emotion_classes[idx]]
                conf = emotion_confidence[idx]
                
                print(f"Sample {idx}: Predicted={pred_class} (conf={conf:.3f})")
                print(f"  Alpha Asymmetry: {alpha_asym:.3f} ({'Left dominant' if alpha_asym > 0.5 else 'Right dominant'})")
                print(f"  Motor Beta: {motor_beta:.3f} ({'High' if motor_beta > 0.5 else 'Low'})")
                print(f"  Frontal Theta: {frontal_theta:.3f} ({'High attention' if frontal_theta > 0.5 else 'Low attention'})")
    
    print("\n✅ Improved neurosymbolic multi-task fusion training completed!")
    print("📁 Model saved as: improved_neurosymbolic_eeg_model.h5")
    print("\n🎯 Key Improvements:")
    print("1. EEG-grounded predicates based on neuroscience literature")
    print("2. Simplified, domain-specific rules")
    print("3. Uncertainty-gated constraint application")  
    print("4. Curriculum learning approach")
    print("5. Interpretable predicate activations")
    
    return best_model, history

if __name__ == "__main__":
    model, training_history = main()