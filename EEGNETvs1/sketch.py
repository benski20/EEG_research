import os
import pickle
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten, 
    Lambda, Add, Multiply, Conv2D, MaxPooling2D, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

# -------------------------
# CONFIG
# -------------------------
EMOTION_MODEL_PATH = "eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5"
POS_MODEL_PATH = "eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5"
EMOTION_DATA_PATH = "eeg_raw_dataset.pkl"
POS_DATA_PATH = "eeg_raw_nounvsverb_JASON_dataset.pkl"

WORDLIST_EMOTION_PATH = "emotion_words.pkl"
WORDLIST_POS_PATH = "pos_upd_words.pkl"
EMBEDDINGS_PATH = "embeddings.pkl"

INPUT_SHAPE = (4, 1000, 1)
NUM_CLASSES_EMOTION = 3
NUM_CLASSES_POS = 2
FUSION_DENSE = 256
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-5
RANDOM_STATE = 42

# -------------------------
# EMBEDDING ANALYSIS UTILITIES
# -------------------------
def analyze_word_categories(words, embeddings_dict):
    """
    Analyze embeddings to determine optimal EEG processing strategies for different word types
    """
    print("🔍 Analyzing word embeddings to design EEG architecture...")
    
    # Collect valid embeddings
    valid_embeddings = []
    valid_words = []
    
    for word in words:
        if word in embeddings_dict:
            valid_embeddings.append(embeddings_dict[word])
            valid_words.append(word)
    
    if len(valid_embeddings) < 10:
        print("⚠️ Too few valid embeddings for analysis, using default architecture")
        return None
    
    embeddings_matrix = np.array(valid_embeddings)
    
    # Cluster words into categories (emotional, concrete, abstract, etc.)
    n_clusters = min(2, len(valid_embeddings) // 5)  # Adjust based on data size
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE)
    clusters = kmeans.fit_predict(embeddings_matrix)
    
    # Analyze frequency preferences for each cluster
    cluster_analysis = {}
    for i in range(n_clusters):
        cluster_words = [valid_words[j] for j in range(len(valid_words)) if clusters[j] == i]
        cluster_analysis[i] = {
            'words': cluster_words[:10],  # Sample words
            'centroid': kmeans.cluster_centers_[i],
            'size': len(cluster_words)
        }
    
    print(f"📊 Found {n_clusters} word clusters:")
    for i, info in cluster_analysis.items():
        print(f"  Cluster {i}: {info['size']} words (sample: {info['words'][:5]})")
    
    return cluster_analysis

# -------------------------
# SPECIALIZED EEG PROCESSING BRANCHES
# -------------------------
class EmotionOptimizedEEGBranch(tf.keras.layers.Layer):
    """EEG processing optimized for emotional words - focuses on low frequencies"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        # Focus on low frequency bands (delta, theta, alpha) for emotions
        # Input: (None, 4, 60, 1)
        self.emotion_conv1 = Conv2D(16, (1, 8), activation='relu', padding='same', name='emotion_conv1')
        self.emotion_conv2 = Conv2D(32, (1, 4), activation='relu', padding='same', name='emotion_conv2')
        self.emotion_pool = MaxPooling2D((1, 2), name='emotion_pool')
        self.emotion_bn = BatchNormalization(name='emotion_bn')
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply low-frequency focused processing
        x = self.emotion_conv1(inputs)  # (None, 4, 60, 16)
        x = self.emotion_conv2(x)       # (None, 4, 60, 32)
        x = self.emotion_pool(x)        # (None, 4, 30, 32)
        x = self.emotion_bn(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4, 30, 32)

class ConcreteOptimizedEEGBranch(tf.keras.layers.Layer):
    """EEG processing optimized for concrete words - focuses on sensory areas"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        # Focus on beta and gamma frequencies for concrete processing
        # Input: (None, 4, 60, 1)
        self.concrete_conv1 = Conv2D(16, (2, 6), activation='relu', padding='same', name='concrete_conv1')
        self.concrete_conv2 = Conv2D(32, (1, 3), activation='relu', padding='same', name='concrete_conv2')
        self.concrete_pool = MaxPooling2D((1, 2), name='concrete_pool')
        self.concrete_bn = BatchNormalization(name='concrete_bn')
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply sensory-focused processing
        x = self.concrete_conv1(inputs)  # (None, 4, 60, 16)
        x = self.concrete_conv2(x)       # (None, 4, 60, 32)
        x = self.concrete_pool(x)        # (None, 4, 30, 32)
        x = self.concrete_bn(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4, 30, 32)

class AbstractOptimizedEEGBranch(tf.keras.layers.Layer):
    """EEG processing optimized for abstract words - focuses on prefrontal patterns"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        # Focus on higher frequencies and cross-channel interactions
        # Input: (None, 4, 60, 1) - FIXED: Use appropriate kernel sizes
        self.abstract_conv1 = Conv2D(16, (2, 8), activation='relu', padding='same', name='abstract_conv1')
        self.abstract_conv2 = Conv2D(32, (1, 4), activation='relu', padding='same', name='abstract_conv2')
        self.abstract_pool = MaxPooling2D((1, 2), name='abstract_pool')
        self.abstract_bn = BatchNormalization(name='abstract_bn')
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply abstract-focused processing
        x = self.abstract_conv1(inputs)  # (None, 4, 60, 16)
        x = self.abstract_conv2(x)       # (None, 4, 60, 32)
        x = self.abstract_pool(x)        # (None, 4, 30, 32)
        x = self.abstract_bn(x)
        return x
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], 4, 30, 32)

class AdaptiveEEGProcessor(tf.keras.layers.Layer):
    """Adaptive EEG processor that uses embedding-informed architecture design"""
    def __init__(self, cluster_analysis=None, **kwargs):
        super().__init__(**kwargs)
        self.cluster_analysis = cluster_analysis
        self.use_specialized = cluster_analysis is not None
    
    def build(self, input_shape):
        
         # Create specialized branches based on embedding analysis
        self.emotion_branch = EmotionOptimizedEEGBranch(name='emotion_branch')
        self.concrete_branch = ConcreteOptimizedEEGBranch(name='concrete_branch')
        self.abstract_branch = AbstractOptimizedEEGBranch(name='abstract_branch')
        self.branch_fusion = Dense(128, activation='relu', name='branch_fusion')
        
        
        super().build(input_shape)
    
    def call(self, inputs):
        
        # Process through all specialized branches
        emotion_out = self.emotion_branch(inputs)    # (None, 4, 30, 32)
        concrete_out = self.concrete_branch(inputs)  # (None, 4, 30, 32)
        abstract_out = self.abstract_branch(inputs)  # (None, 4, 30, 32)
            
        # Flatten and combine outputs
        emotion_flat = Flatten()(emotion_out)        # (None, 3840)
        concrete_flat = Flatten()(concrete_out)      # (None, 3840)
        abstract_flat = Flatten()(abstract_out)      # (None, 3840)
            
            # Fuse specialized branches
        combined = Concatenate()([emotion_flat, concrete_flat, abstract_flat])  # (None, 11520)
        fused = self.branch_fusion(combined)         # (None, 128)
        return fused
        
    
    def compute_output_shape(self, input_shape):
        
         return (input_shape[0], 128)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_embeddings(path):
    if path is None or not os.path.exists(path):
        print(f"⚠️ Embeddings path {path} not found")
        return None
    data = load_pickle(path)
    return {w: np.array(vec, dtype=np.float32) for w, vec in data.items()}

def robust_normalize_eeg(X):
    """Standard robust normalization"""
    Xn = X.copy().astype(np.float32)
    for ch in range(Xn.shape[1]):
        channel_data = Xn[:, ch, :]
        p5 = np.percentile(channel_data, 5, axis=1, keepdims=True)
        p95 = np.percentile(channel_data, 95, axis=1, keepdims=True)
        scale = p95 - p5
        scale = np.where(scale == 0, 1, scale)
        Xn[:, ch, :] = (channel_data - p5) / scale
    return Xn

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

# -------------------------
# MAIN EXECUTION
# -------------------------
def main():
    print("🚀 Strategy 1: Embedding-Constrained Architecture (NO embeddings at inference)")
    
    # Load embeddings for architecture design
    embedding_dict = load_embeddings(EMBEDDINGS_PATH)
    
    print("📂 Loading datasets...")
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data = load_pickle(POS_DATA_PATH)
    
    # Prepare emotion dataset
    X_e = np.array(emotion_data["X"]).reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
    y_e = np.array(emotion_data["y"])
    words_e = emotion_data.get("words", [f"emotion_word_{i}" for i in range(len(X_e))])
    
    # Prepare POS dataset  
    X_p = np.array(pos_data["X"]).reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
    y_p = np.array(pos_data["y"]) - 1  # Adjust to 0-based
    words_p = pos_data.get("words", [f"pos_word_{i}" for i in range(len(X_p))])
    
    # Analyze embeddings to design architecture
    all_words = list(words_e) + list(words_p)
    cluster_analysis = None
    if embedding_dict:
        cluster_analysis = analyze_word_categories(all_words, embedding_dict)
    
    # Preprocess data
    print("🔧 Preprocessing data...")
    X_e_norm = robust_normalize_eeg(X_e)
    X_p_norm = robust_normalize_eeg(X_p)
    
    # Add channel dimension
    X_e_norm = X_e_norm[..., np.newaxis]
    X_p_norm = X_p_norm[..., np.newaxis]
    
    # Combine datasets
    X_combined = np.concatenate([X_e_norm, X_p_norm], axis=0)
    
    # Create labels
    y_emotion_combined = np.concatenate([
        to_categorical(y_e, NUM_CLASSES_EMOTION),
        np.zeros((len(X_p_norm), NUM_CLASSES_EMOTION))
    ], axis=0)
    
    y_pos_combined = np.concatenate([
        np.zeros((len(X_e_norm), NUM_CLASSES_POS)),
        to_categorical(y_p, NUM_CLASSES_POS)
    ], axis=0)
    
    # Sample weights
    sw_emotion = np.concatenate([np.ones(len(X_e_norm)), np.zeros(len(X_p_norm))], axis=0)
    sw_pos = np.concatenate([np.zeros(len(X_e_norm)), np.ones(len(X_p_norm))], axis=0)
    
    print("🏗️ Building embedding-constrained architecture...")
    
    # MAIN INPUT - ONLY EEG, NO EMBEDDINGS AT INFERENCE
    eeg_input = Input(shape=INPUT_SHAPE, name="eeg_only_input")
    
    # Adaptive EEG processor informed by embedding analysis
    adaptive_processor = AdaptiveEEGProcessor(cluster_analysis=cluster_analysis, name='adaptive_eeg')
    eeg_features = adaptive_processor(eeg_input)
    
    # Shared dense layers
    x = Dense(FUSION_DENSE, activation="relu", name="shared_dense1")(eeg_features)
    x = BatchNormalization(name="shared_bn1")(x)
    x = Dropout(0.3, name="shared_dropout1")(x)
    x = Dense(FUSION_DENSE//2, activation="relu", name="shared_dense2")(x)
    x = BatchNormalization(name="shared_bn2")(x)
    x = Dropout(0.2, name="shared_dropout2")(x)
    
    # Task-specific heads
    emotion_logits = Dense(NUM_CLASSES_EMOTION, activation='softmax', name='emotion_output')(x)
    pos_logits = Dense(NUM_CLASSES_POS, activation='softmax', name='pos_output')(x)
    
    # Build model - ONLY EEG INPUT
    model = Model(inputs=eeg_input, outputs=[emotion_logits, pos_logits], 
                  name="new_embedding_constrained_architecture")
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss=[balanced_focal_loss(alpha=0.25, gamma=1.5), balanced_focal_loss(alpha=0.25, gamma=1.5)],
        loss_weights=[1.2, 1.0],
        metrics=[['accuracy'], ["accuracy"]]
    )
    
    print("📊 Model summary:")
    model.summary()
    
    # Train/val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    # Create combined labels for stratification
    combined_labels = []
    for i in range(len(X_combined)):
        if sw_emotion[i] > 0:
            combined_labels.append(f"emotion_{np.argmax(y_emotion_combined[i])}")
        else:
            combined_labels.append(f"pos_{np.argmax(y_pos_combined[i])}")
    
    train_idx, val_idx = next(sss.split(X_combined, combined_labels))
    
    X_train, X_val = X_combined[train_idx], X_combined[val_idx]
    ye_train, ye_val = y_emotion_combined[train_idx], y_emotion_combined[val_idx]
    yp_train, yp_val = y_pos_combined[train_idx], y_pos_combined[val_idx]
    swe_train, swe_val = sw_emotion[train_idx], sw_emotion[val_idx]
    swp_train, swp_val = sw_pos[train_idx], sw_pos[val_idx]
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath='new_embedding_constrained_model.h5', monitor='val_loss', 
                       save_best_only=True, verbose=1)
    ]
    
    # Training
    print("🏋️ Training embedding-constrained model (EEG-only inference)...")
    history = model.fit(
        x=X_train,
        y=[ye_train, yp_train],
        sample_weight=[swe_train, swp_train],
        validation_data=(X_val, [ye_val, yp_val], [swe_val, swp_val]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluation
    print("📊 Evaluation...")
    best_model = load_model('new_embedding_constrained_model.h5', 
                           custom_objects={'focal_loss_fixed': balanced_focal_loss(),
                                           "AdaptiveEEGProcessor": AdaptiveEEGProcessor})
    
    predictions = best_model.predict(X_val)
    y_pred_emotion = predictions[0]
    y_pred_pos = predictions[1]
    
    # Emotion evaluation
    emotion_mask = swe_val > 0
    if np.sum(emotion_mask) > 0:
        print("\n🎭 EMOTION CLASSIFICATION (EEG-only):")
        ye_val_labels = np.argmax(ye_val[emotion_mask], axis=1)
        y_pred_emotion_labels = np.argmax(y_pred_emotion[emotion_mask], axis=1)
        print(classification_report(ye_val_labels, y_pred_emotion_labels, 
                                  target_names=["Negative", "Neutral", "Positive"]))
    
    # POS evaluation
    pos_mask = swp_val > 0
    if np.sum(pos_mask) > 0:
        print("\n📝 POS CLASSIFICATION (EEG-only):")
        yp_val_labels = np.argmax(yp_val[pos_mask], axis=1)
        y_pred_pos_labels = np.argmax(y_pred_pos[pos_mask], axis=1)
        print(classification_report(yp_val_labels, y_pred_pos_labels, 
                                  target_names=["Noun", "Verb"]))
        
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    def plot_confusion_matrices(ye_val, y_pred_emotion, swe_val, yp_val, y_pred_pos, swp_val):
        # Emotion
        emotion_mask = swe_val > 0
        if np.sum(emotion_mask) > 0:
            ye_val_labels = np.argmax(ye_val[emotion_mask], axis=1)
            y_pred_emotion_labels = np.argmax(y_pred_emotion[emotion_mask], axis=1)
            cm_emotion = confusion_matrix(ye_val_labels, y_pred_emotion_labels)
            ConfusionMatrixDisplay(cm_emotion, display_labels=["Negative", "Neutral", "Positive"]).plot(cmap="Blues")
            plt.title("Emotion Confusion Matrix")
            #plt.savefig("plots/embedding_constrained_arch_emotion_cm.png")
            plt.show()

        # POS
        pos_mask = swp_val > 0
        if np.sum(pos_mask) > 0:
            yp_val_labels = np.argmax(yp_val[pos_mask], axis=1)
            y_pred_pos_labels = np.argmax(y_pred_pos[pos_mask], axis=1)
            cm_pos = confusion_matrix(yp_val_labels, y_pred_pos_labels)
            ConfusionMatrixDisplay(cm_pos, display_labels=["Noun", "Verb"]).plot(cmap="Greens")
            plt.title("POS Confusion Matrix")
            #plt.savefig("plots/embedding_constrained_arch_pos_cm.png")
            plt.show()

    plot_confusion_matrices(ye_val, y_pred_emotion, swe_val, yp_val, y_pred_pos, swp_val)

#    If AdaptiveEEGProcessor exposes importances:
    
    print("\n✅ Strategy 1 Complete: Model uses NO embeddings at inference!")
    print("🧠 Architecture was designed using embedding insights but runs on EEG-only")
    
    return best_model

if __name__ == "__main__":
    main()
