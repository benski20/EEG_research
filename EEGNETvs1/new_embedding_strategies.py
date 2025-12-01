# embedding_constrained_architecture.py
# Strategy 1: Use embeddings to design better EEG processing architecture, NO embeddings at inference
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
EMOTION_DATA_PATH = "eeg_fft_dataset.pkl"
POS_DATA_PATH = "eeg_fft_nounvsverb_JASON_dataset.pkl"

WORDLIST_EMOTION_PATH = "emotion_words.pkl"
WORDLIST_POS_PATH = "pos_upd_words.pkl"
EMBEDDINGS_PATH = "embeddings.pkl"

INPUT_SHAPE = (4, 60, 1)
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
        self.emotion_conv1 = Conv2D(16, (1, 8), activation='relu', name='emotion_conv1')
        self.emotion_conv2 = Conv2D(32, (1, 4), activation='relu', name='emotion_conv2')
        self.emotion_pool = MaxPooling2D((1, 2), name='emotion_pool')
        self.emotion_bn = BatchNormalization(name='emotion_bn')
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply low-frequency focused processing
        x = self.emotion_conv1(inputs)
        x = self.emotion_conv2(x)
        x = self.emotion_pool(x)
        x = self.emotion_bn(x)
        return x

class ConcreteOptimizedEEGBranch(tf.keras.layers.Layer):
    """EEG processing optimized for concrete words - focuses on sensory areas"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        # Focus on beta and gamma frequencies for concrete processing
        self.concrete_conv1 = Conv2D(16, (2, 6), activation='relu', name='concrete_conv1')
        self.concrete_conv2 = Conv2D(32, (1, 3), activation='relu', name='concrete_conv2')
        self.concrete_pool = MaxPooling2D((1, 2), name='concrete_pool')
        self.concrete_bn = BatchNormalization(name='concrete_bn')
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply sensory-focused processing
        x = self.concrete_conv1(inputs)
        x = self.concrete_conv2(x)
        x = self.concrete_pool(x)
        x = self.concrete_bn(x)
        return x

class AbstractOptimizedEEGBranch(tf.keras.layers.Layer):
    """EEG processing optimized for abstract words - focuses on prefrontal patterns"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        # Focus on higher frequencies and cross-channel interactions
        self.abstract_conv1 = Conv2D(16, (4, 4), activation='relu', name='abstract_conv1')
        self.abstract_conv2 = Conv2D(32, (2, 6), activation='relu', name='abstract_conv2')
        self.abstract_pool = MaxPooling2D((1, 2), name='abstract_pool')
        self.abstract_bn = BatchNormalization(name='abstract_bn')
        super().build(input_shape)
    
    def call(self, inputs):
        # Apply abstract-focused processing
        x = self.abstract_conv1(inputs)
        x = self.abstract_conv2(x)
        x = self.abstract_pool(x)
        x = self.abstract_bn(x)
        return x

class AdaptiveEEGProcessor(tf.keras.layers.Layer):
    """Adaptive EEG processor that uses embedding-informed architecture design"""
    def __init__(self, cluster_analysis=None, **kwargs):
        super().__init__(**kwargs)
        self.cluster_analysis = cluster_analysis
        self.use_specialized = cluster_analysis is not None
    
    def build(self, input_shape):
        if self.use_specialized:
            # Create specialized branches based on embedding analysis
            self.emotion_branch = EmotionOptimizedEEGBranch(name='emotion_branch')
            self.concrete_branch = ConcreteOptimizedEEGBranch(name='concrete_branch')
            self.abstract_branch = AbstractOptimizedEEGBranch(name='abstract_branch')
            self.branch_fusion = Dense(128, activation='relu', name='branch_fusion')
        else:
            # Fallback to general processing
            self.general_conv1 = Conv2D(32, (2, 8), activation='relu', name='general_conv1')
            self.general_conv2 = Conv2D(64, (1, 4), activation='relu', name='general_conv2')
            self.general_pool = MaxPooling2D((1, 2), name='general_pool')
            self.general_bn = BatchNormalization(name='general_bn')
        
        self.global_pool = GlobalAveragePooling2D(name='adaptive_global_pool')
        super().build(input_shape)
    
    def call(self, inputs):
        if self.use_specialized:
            # Process through all specialized branches
            emotion_out = self.emotion_branch(inputs)
            concrete_out = self.concrete_branch(inputs)
            abstract_out = self.abstract_branch(inputs)
            
            # Flatten and combine outputs
            emotion_flat = Flatten()(emotion_out)
            concrete_flat = Flatten()(concrete_out)
            abstract_flat = Flatten()(abstract_out)
            
            # Fuse specialized branches
            combined = Concatenate()([emotion_flat, concrete_flat, abstract_flat])
            fused = self.branch_fusion(combined)
            return fused
        else:
            # General processing
            x = self.general_conv1(inputs)
            x = self.general_conv2(x)
            x = self.general_pool(x)
            x = self.general_bn(x)
            return self.global_pool(x)

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
    X_e = np.array(emotion_data["data"]).reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
    y_e = np.array(emotion_data["labels"])
    words_e = emotion_data.get("words", [f"emotion_word_{i}" for i in range(len(X_e))])
    
    # Prepare POS dataset  
    X_p = np.array(pos_data["data"]).reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
    y_p = np.array(pos_data["labels"]) - 1  # Adjust to 0-based
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
                  name="embedding_constrained_architecture")
    
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
        ModelCheckpoint(filepath='upd_embedding_constrained_model.h5', monitor='val_loss', 
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
    best_model = load_model('upd_embedding_constrained_model.h5', 
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


# embedding_guided_augmentation.py  
# # Strategy 3: Use embeddings only for creating better training data, NO embeddings at inference
# import os
# import pickle
# import hashlib
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import (
#     Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten, 
#     Lambda, Add, Multiply
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, confusion_matrix, pairwise_distances
# from sklearn.neighbors import NearestNeighbors
# from imblearn.over_sampling import SMOTE
# import matplotlib.pyplot as plt
# import tensorflow.keras.backend as K

# # -------------------------
# # CONFIG
# # -------------------------
# EMOTION_MODEL_PATH = "eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5"
# POS_MODEL_PATH = "eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5"
# EMOTION_DATA_PATH = "eeg_fft_dataset.pkl"
# POS_DATA_PATH = "eeg_fft_nounvsverb_JASON_dataset.pkl"

# WORDLIST_EMOTION_PATH = "emotion_words.pkl"
# WORDLIST_POS_PATH = "pos_upd_words.pkl"
# EMBEDDINGS_PATH = "embeddings.pkl"

# INPUT_SHAPE = (4, 60, 1)
# NUM_CLASSES_EMOTION = 3
# NUM_CLASSES_POS = 2
# FUSION_DENSE = 256
# BATCH_SIZE = 32
# EPOCHS = 100
# LR = 1e-5
# RANDOM_STATE = 42
# EMB_DIM = 64

# # Augmentation parameters
# AUG_MULTIPLIER = 2  # How many augmented samples per original
# SIMILARITY_THRESHOLD = 0.7  # Minimum similarity for semantic augmentation
# MAX_AUG_STRENGTH = 0.3  # Maximum augmentation strength

# # -------------------------
# # EMBEDDING-GUIDED AUGMENTATION
# # -------------------------
# def find_similar_words(target_word, embeddings_dict, top_k=5, min_similarity=0.3):
#     """
#     Find semantically similar words based on embedding cosine similarity
#     """
#     if target_word not in embeddings_dict:
#         return []
    
#     target_emb = embeddings_dict[target_word]
#     similarities = []
    
#     for word, emb in embeddings_dict.items():
#         if word != target_word:
#             # Compute cosine similarity
#             sim = np.dot(target_emb, emb) / (np.linalg.norm(target_emb) * np.linalg.norm(emb))
#             if sim >= min_similarity:
#                 similarities.append((word, sim))
    
#     # Sort by similarity and return top_k
#     similarities.sort(key=lambda x: x[1], reverse=True)
#     return similarities[:top_k]

# def semantic_eeg_augmentation(eeg_sample, similarity_score, base_strength=0.1):
#     """
#     Apply EEG augmentation based on semantic similarity
#     Higher similarity = lighter augmentation (words are more similar)
#     Lower similarity = stronger augmentation (words are more different)
#     """
#     # Inverse relationship: more similar words need less augmentation
#     aug_strength = base_strength * (1.0 - similarity_score)
#     aug_strength = np.clip(aug_strength, 0.01, MAX_AUG_STRENGTH)
    
#     augmented = eeg_sample.copy()
    
#     # Frequency domain augmentation
#     for ch in range(augmented.shape[0]):  # For each channel
#         channel_data = augmented[ch, :]
        
#         # Add semantic-guided noise
#         noise_std = aug_strength * np.std(channel_data)
#         semantic_noise = np.random.normal(0, noise_std, channel_data.shape)
        
#         # Frequency-specific augmentation based on similarity
#         # More similar words: preserve low frequencies (semantic core)
#         # Less similar words: allow more variation across all frequencies
#         freq_mask = np.ones_like(channel_data)
#         if similarity_score > 0.7:  # Very similar words
#             # Preserve low frequencies (0-20 bins), augment high frequencies
#             freq_mask[:20] *= 0.3  # Light augmentation for low freq
#             freq_mask[20:] *= 1.0  # Normal augmentation for high freq
#         elif similarity_score > 0.4:  # Moderately similar
#             # Moderate augmentation across all frequencies
#             freq_mask *= 0.7
#         else:  # Less similar words
#             # Allow stronger augmentation
#             freq_mask *= 1.2
        
#         augmented[ch, :] = channel_data + semantic_noise * freq_mask
    
#     return augmented

# def embedding_guided_data_augmentation(X_eeg, words, labels, embeddings_dict, multiplier=AUG_MULTIPLIER):
#     """
#     Create augmented training data using semantic embedding information
#     """
#     print(f"🎨 Creating embedding-guided augmented data (multiplier: {multiplier})...")
    
#     if embeddings_dict is None:
#         print("⚠️ No embeddings available, using standard augmentation")
#         return standard_augmentation(X_eeg, labels, multiplier)
    
#     augmented_samples = []
#     augmented_labels = []
#     augmented_words = []
    
#     # Track augmentation statistics
#     aug_stats = {'semantic_pairs': 0, 'fallback_aug': 0}
    
#     for i, (eeg_sample, word, label) in enumerate(zip(X_eeg, words, labels)):
#         # Keep original sample
#         augmented_samples.append(eeg_sample)
#         augmented_labels.append(label)
#         augmented_words.append(word)
        
#         # Find semantically similar words
#         similar_words = find_similar_words(word, embeddings_dict, 
#                                          top_k=multiplier, 
#                                          min_similarity=SIMILARITY_THRESHOLD)
        
#         if similar_words:
#             # Create semantically-informed augmented samples
#             for similar_word, similarity in similar_words:
#                 aug_sample = semantic_eeg_augmentation(eeg_sample, similarity)
#                 augmented_samples.append(aug_sample)
#                 augmented_labels.append(label)  # Same label, different word
#                 augmented_words.append(f"{word}~{similar_word}")  # Track relationship
#                 aug_stats['semantic_pairs'] += 1
#         else:
#             # Fallback to standard augmentation if no similar words found
#             for _ in range(multiplier):
#                 aug_sample = standard_eeg_augmentation(eeg_sample, strength=0.15)
#                 augmented_samples.append(aug_sample)
#                 augmented_labels.append(label)
#                 augmented_words.append(f"{word}~aug")
#                 aug_stats['fallback_aug'] += 1
    
#     print(f"📊 Augmentation stats:")
#     print(f"  - Semantic pairs: {aug_stats['semantic_pairs']}")
#     print(f"  - Fallback augmentations: {aug_stats['fallback_aug']}")
#     print(f"  - Total samples: {len(augmented_samples)} (from {len(X_eeg)})")
    
#     return np.array(augmented_samples), np.array(augmented_labels), augmented_words

# def standard_eeg_augmentation(eeg_sample, strength=0.1):
#     """Standard EEG augmentation without semantic guidance"""
#     augmented = eeg_sample.copy()
    
#     # Add noise
#     noise_std = strength * np.std(augmented)
#     noise = np.random.normal(0, noise_std, augmented.shape)
    
#     # Slight frequency shifting
#     shift_amount = int(strength * 5)  # Small frequency shift
#     if shift_amount > 0 and augmented.shape[1] > shift_amount:
#         augmented[:, :-shift_amount] = augmented[:, shift_amount:]
    
#     return augmented + noise

# def standard_augmentation(X_eeg, labels, multiplier):
#     """Fallback standard augmentation when no embeddings available"""
#     augmented_samples = []
#     augmented_labels = []
    
#     for eeg_sample, label in zip(X_eeg, labels):
#         augmented_samples.append(eeg_sample)
#         augmented_labels.append(label)
        
#         for _ in range(multiplier):
#             aug_sample = standard_eeg_augmentation(eeg_sample)
#             augmented_samples.append(aug_sample)
#             augmented_labels.append(label)
    
#     return np.array(augmented_samples), np.array(augmented_labels), None

# def balanced_sampling_with_embeddings(X_eeg, words, labels, embeddings_dict, target_samples_per_class=None):
#     """
#     Create balanced dataset using embedding-informed sampling
#     """
#     print("⚖️ Creating balanced dataset with embedding-guided sampling...")
    
#     unique_labels, counts = np.unique(labels, return_counts=True)
    
#     if target_samples_per_class is None:
#         target_samples_per_class = max(counts) + 100  # Slightly oversample
    
#     print(f"📊 Original class distribution: {dict(zip(unique_labels, counts))}")
#     print(f"🎯 Target samples per class: {target_samples_per_class}")
    
#     balanced_samples = []
#     balanced_labels = []
#     balanced_words = []
    
#     for class_label in unique_labels:
#         class_mask = (labels == class_label)
#         class_samples = X_eeg[class_mask]
#         class_words = [words[i] for i in range(len(words)) if class_mask[i]]
        
#         current_count = len(class_samples)
        
#         if current_count >= target_samples_per_class:
#             # Randomly sample down
#             indices = np.random.choice(current_count, target_samples_per_class, replace=False)
#             selected_samples = class_samples[indices]
#             selected_words = [class_words[i] for i in indices]
#         else:
#             # Need to augment
#             selected_samples = list(class_samples)
#             selected_words = list(class_words)
            
#             needed = target_samples_per_class - current_count
#             print(f"  Class {class_label}: need {needed} more samples")
            
#             # Use embedding-guided augmentation to reach target
#             for _ in range(needed):
#                 # Pick random sample from this class
#                 base_idx = np.random.randint(0, current_count)
#                 base_sample = class_samples[base_idx]
#                 base_word = class_words[base_idx]
                
#                 # Find similar word and augment accordingly
#                 if embeddings_dict and base_word in embeddings_dict:
#                     similar_words = find_similar_words(base_word, embeddings_dict, top_k=3)
#                     if similar_words:
#                         _, similarity = similar_words[0]  # Use most similar
#                         aug_sample = semantic_eeg_augmentation(base_sample, similarity)
#                     else:
#                         aug_sample = standard_eeg_augmentation(base_sample)
#                 else:
#                     aug_sample = standard_eeg_augmentation(base_sample)
                
#                 selected_samples.append(aug_sample)
#                 selected_words.append(f"{base_word}~bal")
        
#         # Add to balanced dataset
#         balanced_samples.extend(selected_samples)
#         balanced_labels.extend([class_label] * len(selected_samples))
#         balanced_words.extend(selected_words)
    
#     final_samples = np.array(balanced_samples)
#     final_labels = np.array(balanced_labels)
    
#     # Verify balance
#     final_unique, final_counts = np.unique(final_labels, return_counts=True)
#     print(f"📊 Final balanced distribution: {dict(zip(final_unique, final_counts))}")
    
#     return final_samples, final_labels, balanced_words

# # -------------------------
# # HELPER FUNCTIONS  
# # -------------------------
# def load_pickle(path):
#     with open(path, "rb") as f:
#         return pickle.load(f)

# def get_feature_extractor(model):
#     """Extract features before final classification layer"""
#     flatten_layer = None
#     for layer in model.layers:
#         if "flatten" in layer.name.lower():
#             flatten_layer = layer
#             break
#     if flatten_layer is not None:
#         return Model(inputs=model.input, outputs=flatten_layer.output)
#     else:
#         if len(model.layers) >= 2:
#             return Model(inputs=model.input, outputs=model.layers[-2].output)
#         else:
#             raise ValueError("Could not identify a feature layer in model.")

# def load_embeddings(path):
#     if path is None or not os.path.exists(path):
#         print(f"⚠️ Embeddings path {path} not found")
#         return None
#     data = load_pickle(path)
#     return {w: np.array(vec, dtype=np.float32) for w, vec in data.items()}

# def robust_normalize_eeg(X):
#     """Standard robust normalization"""
#     Xn = X.copy().astype(np.float32)
#     for ch in range(Xn.shape[1]):
#         channel_data = Xn[:, ch, :]
#         p5 = np.percentile(channel_data, 5, axis=1, keepdims=True)
#         p95 = np.percentile(channel_data, 95, axis=1, keepdims=True)
#         scale = p95 - p5
#         scale = np.where(scale == 0, 1, scale)
#         Xn[:, ch, :] = (channel_data - p5) / scale
#     return Xn

# def balanced_focal_loss(alpha=0.25, gamma=1.5):
#     def focal_loss_fixed(y_true, y_pred):
#         epsilon = tf.keras.backend.epsilon()
#         y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
#         ce = -y_true * tf.math.log(y_pred)
#         alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
#         p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
#         focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
#         focal_loss = focal_weight * ce
#         return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
#     return focal_loss_fixed

# def create_simple_attention_fusion(emotion_features, pos_features, fusion_dim=256):
#     """Simple attention-based fusion"""
#     emotion_proj = Dense(fusion_dim//2, activation='relu', name='emotion_proj')(emotion_features)
#     pos_proj = Dense(fusion_dim//2, activation='relu', name='pos_proj')(pos_features)
    
#     # Simple attention
#     emotion_attention = Dense(1, activation='sigmoid', name='emotion_att')(emotion_proj)
#     pos_attention = Dense(1, activation='sigmoid', name='pos_att')(pos_proj)
    
#     emotion_weighted = Lambda(lambda x: x[0] * x[1])([emotion_proj, emotion_attention])
#     pos_weighted = Lambda(lambda x: x[0] * x[1])([pos_proj, pos_attention])
#     fused = Concatenate()([emotion_weighted, pos_weighted])
#     return fused

# # -------------------------
# # MAIN EXECUTION
# # -------------------------
# def main():
#     print("🚀 Strategy 3: Embedding-Guided Data Augmentation")
#     print("🎨 Using embeddings ONLY for creating better training data")
#     print("🧠 NO embeddings used at inference - pure EEG classification!")
    
#     # Load embeddings for augmentation design
#     embedding_dict = load_embeddings(EMBEDDINGS_PATH)
    
#     # Load pretrained feature extractors
#     print("📦 Loading pretrained models...")
#     emotion_model = load_model(EMOTION_MODEL_PATH)
#     pos_model = load_model(POS_MODEL_PATH)
    
#     emotion_extractor = get_feature_extractor(emotion_model)
#     pos_extractor = get_feature_extractor(pos_model)
#     emotion_extractor.trainable = False
#     pos_extractor.trainable = False
    
#     # Load data
#     print("📂 Loading datasets...")
#     emotion_data = load_pickle(EMOTION_DATA_PATH)
#     pos_data = load_pickle(POS_DATA_PATH)
    
#     # Prepare emotion dataset
#     X_e = np.array(emotion_data["data"]).reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
#     y_e = np.array(emotion_data["labels"])
#     words_e = emotion_data.get("words", [f"emotion_word_{i}" for i in range(len(X_e))])
    
#     # Prepare POS dataset  
#     X_p = np.array(pos_data["data"]).reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
#     y_p = np.array(pos_data["labels"]) - 1  # Adjust to 0-based
#     words_p = pos_data.get("words", [f"pos_word_{i}" for i in range(len(X_p))])
    
#     # Preprocess data
#     print("🔧 Basic preprocessing...")
#     X_e_norm = robust_normalize_eeg(X_e)
#     X_p_norm = robust_normalize_eeg(X_p)
    
#     # EMBEDDING-GUIDED AUGMENTATION for emotion data
#     print("🎨 Applying embedding-guided augmentation to emotion data...")
#     X_e_aug, y_e_aug, words_e_aug = embedding_guided_data_augmentation(
#         X_e_norm, words_e, y_e, embedding_dict, multiplier=AUG_MULTIPLIER
#     )
    
#     # EMBEDDING-GUIDED AUGMENTATION for POS data
#     print("🎨 Applying embedding-guided augmentation to POS data...")
#     X_p_aug, y_p_aug, words_p_aug = embedding_guided_data_augmentation(
#         X_p_norm, words_p, y_p, embedding_dict, multiplier=AUG_MULTIPLIER
#     )
    
#     # BALANCED SAMPLING using embeddings
#     print("⚖️ Creating balanced datasets using embedding-guided sampling...")
#     X_e_balanced, y_e_balanced, words_e_balanced = balanced_sampling_with_embeddings(
#         X_e_aug, words_e_aug if words_e_aug else words_e, y_e_aug, embedding_dict
#     )
    
#     X_p_balanced, y_p_balanced, words_p_balanced = balanced_sampling_with_embeddings(
#         X_p_aug, words_p_aug if words_p_aug else words_p, y_p_aug, embedding_dict
#     )
    
#     print(f"📊 Final augmented dataset sizes:")
#     print(f"  Emotion: {X_e.shape} -> {X_e_balanced.shape}")
#     print(f"  POS: {X_p.shape} -> {X_p_balanced.shape}")
    
#     # Add channel dimension
#     X_e_balanced = X_e_balanced[..., np.newaxis]
#     X_p_balanced = X_p_balanced[..., np.newaxis]
    
#     # Combine datasets
#     X_combined = np.concatenate([X_e_balanced, X_p_balanced], axis=0)
    
#     # Create labels
#     y_emotion_combined = np.concatenate([
#         to_categorical(y_e_balanced, NUM_CLASSES_EMOTION),
#         np.zeros((len(X_p_balanced), NUM_CLASSES_EMOTION))
#     ], axis=0)
    
#     y_pos_combined = np.concatenate([
#         np.zeros((len(X_e_balanced), NUM_CLASSES_POS)),
#         to_categorical(y_p_balanced, NUM_CLASSES_POS)
#     ], axis=0)
    
#     # Sample weights
#     sw_emotion = np.concatenate([np.ones(len(X_e_balanced)), np.zeros(len(X_p_balanced))], axis=0)
#     sw_pos = np.concatenate([np.zeros(len(X_e_balanced)), np.ones(len(X_p_balanced))], axis=0)
    
#     print("🏗️ Building EEG-only model (no embeddings at inference)...")
    
#     # MAIN INPUT - ONLY EEG DATA
#     main_input = Input(shape=INPUT_SHAPE, name="eeg_only_input")
    
#     # Extract features using pretrained extractors
#     feat_a_tensor = emotion_extractor(main_input)
#     feat_b_tensor = pos_extractor(main_input)
#     feat_a_flat = Flatten(name="emotion_features_flat")(feat_a_tensor)
#     feat_b_flat = Flatten(name="pos_features_flat")(feat_b_tensor)
    
#     # Fuse features with attention
#     fused_features = create_simple_attention_fusion(feat_a_flat, feat_b_flat, fusion_dim=FUSION_DENSE)
    
#     # Shared processing layers
#     x = Dense(FUSION_DENSE, activation="relu", name="fusion_dense1")(fused_features)
#     x = BatchNormalization(name="fusion_bn1")(x)
#     x = Dropout(0.3, name="fusion_dropout1")(x)
#     x = Dense(FUSION_DENSE//2, activation="relu", name="fusion_dense2")(x)
#     x = BatchNormalization(name="fusion_bn2")(x)
#     x = Dropout(0.2, name="fusion_dropout2")(x)
    
#     # Task-specific heads
#     emotion_output = Dense(NUM_CLASSES_EMOTION, activation='softmax', name='emotion')(x)
#     pos_output = Dense(NUM_CLASSES_POS, activation='softmax', name='pos')(x)
    
#     # Build model - NO EMBEDDING INPUTS
#     model = Model(inputs=main_input, outputs=[emotion_output, pos_output], 
#                   name="embedding_augmented_eeg_only_model")
    
#     # Compile
#     model.compile(
#         optimizer=Adam(learning_rate=LR),
#         loss=[balanced_focal_loss(alpha=0.25, gamma=1.5), balanced_focal_loss(alpha=0.25, gamma=1.5)],
#         loss_weights=[1.2, 1.0],
#         metrics=[['accuracy'], ["accuracy"]]
#     )
    
#     print("📊 Model summary:")
#     model.summary()
    
#     # Train/val split
#     print("🎯 Creating train/validation split...")
#     sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    
#     # Create combined labels for stratification
#     combined_labels = []
#     for i in range(len(X_combined)):
#         if sw_emotion[i] > 0:
#             combined_labels.append(f"emotion_{np.argmax(y_emotion_combined[i])}")
#         else:
#             combined_labels.append(f"pos_{np.argmax(y_pos_combined[i])}")
    
#     train_idx, val_idx = next(sss.split(X_combined, combined_labels))
    
#     X_train, X_val = X_combined[train_idx], X_combined[val_idx]
#     ye_train, ye_val = y_emotion_combined[train_idx], y_emotion_combined[val_idx]
#     yp_train, yp_val = y_pos_combined[train_idx], y_pos_combined[val_idx]
#     swe_train, swe_val = sw_emotion[train_idx], sw_emotion[val_idx]
#     swp_train, swp_val = sw_pos[train_idx], sw_pos[val_idx]
    
#     # Callbacks
#     callbacks = [
#         EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
#         ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
#         ModelCheckpoint(filepath='embedding_augmented_eeg_model.h5', monitor='val_loss', 
#                        save_best_only=True, verbose=1)
#     ]
    
#     # Training
#     print("🏋️ Training EEG-only model with embedding-augmented data...")
#     print("🎨 Model benefits from embedding-guided augmentation during training")
#     print("🧠 But uses NO embeddings during inference - pure EEG classification!")
    
#     history = model.fit(
#         x=X_train,
#         y=[ye_train, yp_train],
#         sample_weight=[swe_train, swp_train],
#         validation_data=(X_val, [ye_val, yp_val], [swe_val, swp_val]),
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     # Evaluation
#     print("📊 Final evaluation...")
#     best_model = load_model('embedding_augmented_eeg_model.h5', 
#                            custom_objects={'focal_loss_fixed': balanced_focal_loss()})
    
#     # Pure EEG inference
#     predictions = best_model.predict(X_val)
#     y_pred_emotion = predictions[0]
#     y_pred_pos = predictions[1]
    
#     # Emotion evaluation
#     emotion_mask = swe_val > 0
#     if np.sum(emotion_mask) > 0:
#         print("\n🎭 EMOTION CLASSIFICATION (Pure EEG, Embedding-Augmented Training):")
#         ye_val_labels = np.argmax(ye_val[emotion_mask], axis=1)
#         y_pred_emotion_labels = np.argmax(y_pred_emotion[emotion_mask], axis=1)
#         print(classification_report(ye_val_labels, y_pred_emotion_labels, 
#                                   target_names=["Negative", "Neutral", "Positive"]))
        
#         emotion_cm = confusion_matrix(ye_val_labels, y_pred_emotion_labels)
#         print("Confusion Matrix:")
#         print(emotion_cm)
    
#     # POS evaluation
#     pos_mask = swp_val > 0
#     if np.sum(pos_mask) > 0:
#         print("\n📝 POS CLASSIFICATION (Pure EEG, Embedding-Augmented Training):")
#         yp_val_labels = np.argmax(yp_val[pos_mask], axis=1)
#         y_pred_pos_labels = np.argmax(y_pred_pos[pos_mask], axis=1)
#         print(classification_report(yp_val_labels, y_pred_pos_labels, 
#                                   target_names=["Noun", "Verb"]))
    
#     # Show augmentation effectiveness
#     print("\n📈 AUGMENTATION EFFECTIVENESS SUMMARY:")
#     print("🎨 Embeddings were used to:")
#     print("  1. Find semantically similar words for targeted augmentation")
#     print("  2. Apply similarity-based EEG augmentation strength") 
#     print("  3. Create balanced datasets with semantic awareness")
#     print("  4. Generate more diverse and realistic training samples")
#     print("\n🧠 At inference time:")
#     print("  ✅ Model uses ONLY EEG data")
#     print("  ✅ No embedding dependency")
#     print("  ✅ No cheating through embedding shortcuts")
#     print("  ✅ Pure neural signal classification")
    
#     # Save final model
#     best_model.save("embedding_augmented_pure_eeg_model.h5")
#     print("\n✅ Strategy 3 Complete: Embedding-Guided Data Augmentation")
#     print("🎯 Model trained with embedding-enhanced data but runs on pure EEG!")
    
#     return best_model

# if __name__ == "__main__":
#     main()





# # Enhanced Strategy: Multi-Modal Embeddings + Multi-Frequency Architecture + Joint-Embedding-Space
# # embedding_constrained_architecture_enhanced_trainable.py
# # embedding_constrained_architecture_enhanced_trainable.py
# # embedding_constrained_architecture_enhanced_trainable.py
# # embedding_constrained_architecture_enhanced_trainable_FIXED.py
# # Enhanced Strategy: Multi-Modal Embeddings + Multi-Frequency Architecture - TRAINABLE VERSION
# import os
# import pickle
# import hashlib
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Model, load_model
# from tensorflow.keras.layers import (
#     Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten, 
#     Lambda, Add, Multiply, Conv2D, MaxPooling2D, GlobalAveragePooling2D
# )
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
# from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import tensorflow.keras.backend as K
# import warnings
# warnings.filterwarnings('ignore')

# # -------------------------
# # CONFIG
# # -------------------------
# EMOTION_MODEL_PATH = "eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5"
# POS_MODEL_PATH = "eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5"
# EMOTION_DATA_PATH = "eeg_fft_dataset.pkl"
# POS_DATA_PATH = "eeg_fft_nounvsverb_JASON_dataset.pkl"

# WORDLIST_EMOTION_PATH = "emotion_words.pkl"
# WORDLIST_POS_PATH = "pos_upd_words.pkl"
# EMBEDDINGS_PATH = "embeddings.pkl"

# INPUT_SHAPE = (4, 60, 1)
# NUM_CLASSES_EMOTION = 3
# NUM_CLASSES_POS = 2
# FUSION_DENSE = 256
# BATCH_SIZE = 32
# EPOCHS = 50  # Reduced for initial training
# LR = 1e-4  # Slightly higher for better initial training
# RANDOM_STATE = 42

# # Set random seeds for reproducibility
# np.random.seed(RANDOM_STATE)
# tf.random.set_seed(RANDOM_STATE)

# # -------------------------
# # ENHANCED EMBEDDING ANALYSIS 
# # -------------------------
# class MultiModalEmbeddingAnalyzer:
#     """
#     Creates joint word-EEG embedding space and automatically generates 
#     optimal multi-frequency filter architectures
#     """
    
#     def __init__(self, word_embeddings, eeg_data, words, labels):
#         self.word_embeddings = word_embeddings
#         self.eeg_data = eeg_data
#         self.words = words
#         self.labels = labels
#         self.joint_space = None
#         self.frequency_mappings = None
        
#     def create_joint_embedding_space(self):
#         """
#         Revolutionary: Create embeddings that predict both semantic AND neural similarities
#         """
#         print("🧠 Creating joint word-EEG embedding space...")
        
#         try:
#             # Step 1: Extract EEG features for each word
#             eeg_features = self._extract_eeg_features()
            
#             # Step 2: Align word embeddings with EEG features  
#             word_features = self._align_word_embeddings()
            
#             # Step 3: Create joint representation
#             self.joint_space = self._create_joint_space(word_features, eeg_features)
            
#             # Step 4: Analyze joint space for architectural insights
#             self.frequency_mappings = self._analyze_frequency_patterns()
            
#             return self.joint_space
            
#         except Exception as e:
#             print(f"⚠️ Error in embedding analysis: {e}")
#             print("📋 Falling back to default frequency mappings...")
#             return self._create_default_frequency_mappings()
    
#     def _extract_eeg_features(self):
#         """Extract meaningful features from EEG data for each word"""
#         eeg_features = []
        
#         for i in range(min(len(self.words), len(self.eeg_data))):
#             eeg_sample = self.eeg_data[i]  # Shape: (4, 60)
            
#             # Extract frequency-domain features
#             freq_features = self._compute_frequency_features(eeg_sample)
            
#             # Extract spatial features  
#             spatial_features = self._compute_spatial_features(eeg_sample)
            
#             # Extract temporal features
#             temporal_features = self._compute_temporal_features(eeg_sample)
            
#             combined = np.concatenate([freq_features, spatial_features, temporal_features])
#             eeg_features.append(combined)
        
#         return np.array(eeg_features)
    
#     def _compute_frequency_features(self, eeg_sample):
#         """Extract features from different frequency bands"""
#         features = []
        
#         try:
#             # Handle potential shape issues
#             if len(eeg_sample.shape) == 3:
#                 eeg_sample = eeg_sample.squeeze()
            
#             # Ensure we have the right shape
#             if eeg_sample.shape[0] != 4 or eeg_sample.shape[1] != 60:
#                 # Pad or truncate as needed
#                 padded_sample = np.zeros((4, 60))
#                 min_channels = min(4, eeg_sample.shape[0])
#                 min_time = min(60, eeg_sample.shape[1])
#                 padded_sample[:min_channels, :min_time] = eeg_sample[:min_channels, :min_time]
#                 eeg_sample = padded_sample
            
#             # Handle NaN/inf values
#             eeg_sample = np.nan_to_num(eeg_sample, nan=0.0, posinf=0.0, neginf=0.0)
            
#             # Apply FFT and extract frequency band features
#             fft_data = np.fft.fft(eeg_sample, axis=1)
#             fft_mag = np.abs(fft_data)
            
#             # Replace any remaining NaN/inf values after FFT
#             fft_mag = np.nan_to_num(fft_mag, nan=0.0, posinf=0.0, neginf=0.0)
            
#             # Define frequency bands (indices for 60-point FFT)
#             bands = {
#                 'delta': (1, 3),   # Very low frequencies
#                 'theta': (3, 6),   # Low frequencies  
#                 'alpha': (6, 10),  # Mid frequencies
#                 'beta': (10, 20),  # Higher frequencies
#                 'gamma': (20, 30)  # High frequencies
#             }
            
#             for band_name, (start, end) in bands.items():
#                 if end <= fft_mag.shape[1]:
#                     band_power = np.mean(fft_mag[:, start:end], axis=1)
#                     # Ensure no NaN values
#                     band_power = np.nan_to_num(band_power, nan=0.0)
#                     features.extend(band_power)
#                 else:
#                     # Pad with zeros if frequency range is out of bounds
#                     features.extend([0.0] * 4)
            
#             result = np.array(features, dtype=np.float32)
            
#             # Final safety check
#             result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
#             return result
            
#         except Exception as e:
#             print(f"⚠️ Error in frequency feature computation: {e}")
#             # Return safe default features
#             return np.zeros(20, dtype=np.float32)  # 5 bands * 4 channels
    
#     def _compute_spatial_features(self, eeg_sample):
#         """Extract spatial coordination patterns"""
#         try:
#             correlations = []
            
#             # Ensure proper shape
#             if len(eeg_sample.shape) == 3:
#                 eeg_sample = eeg_sample.squeeze()
                
#             # Handle NaN/inf values
#             eeg_sample = np.nan_to_num(eeg_sample, nan=0.0, posinf=0.0, neginf=0.0)
            
#             # Cross-channel correlations
#             n_channels = min(4, eeg_sample.shape[0])
            
#             for i in range(n_channels):
#                 for j in range(i+1, n_channels):
#                     try:
#                         channel_i = eeg_sample[i]
#                         channel_j = eeg_sample[j]
                        
#                         # Check for constant channels
#                         if np.std(channel_i) == 0 or np.std(channel_j) == 0:
#                             corr = 0.0
#                         else:
#                             corr_matrix = np.corrcoef(channel_i, channel_j)
#                             corr = corr_matrix[0, 1]
                            
#                             # Handle NaN correlation
#                             if np.isnan(corr) or np.isinf(corr):
#                                 corr = 0.0
                                
#                         correlations.append(corr)
                        
#                     except Exception as e:
#                         correlations.append(0.0)
            
#             # Pad to ensure consistent length (6 pairs for 4 channels)
#             while len(correlations) < 6:
#                 correlations.append(0.0)
            
#             result = np.array(correlations[:6], dtype=np.float32)
            
#             # Final safety check
#             result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
#             return result
            
#         except Exception as e:
#             print(f"⚠️ Error in spatial feature computation: {e}")
#             return np.zeros(6, dtype=np.float32)
    
#     def _compute_temporal_features(self, eeg_sample):
#         """Extract temporal dynamics"""
#         try:
#             temp_features = []
            
#             # Ensure proper shape
#             if len(eeg_sample.shape) == 3:
#                 eeg_sample = eeg_sample.squeeze()
            
#             # Handle NaN/inf values
#             eeg_sample = np.nan_to_num(eeg_sample, nan=0.0, posinf=0.0, neginf=0.0)
            
#             n_channels = min(4, eeg_sample.shape[0])
            
#             for i in range(n_channels):
#                 channel = eeg_sample[i]
                
#                 # Variance over time
#                 var_val = np.var(channel)
#                 if np.isnan(var_val) or np.isinf(var_val):
#                     var_val = 0.0
#                 temp_features.append(var_val)
                
#                 # Autocorrelation at lag 1 (if possible)
#                 if len(channel) > 1:
#                     try:
#                         # Check for constant channel
#                         if np.std(channel[:-1]) == 0 or np.std(channel[1:]) == 0:
#                             autocorr = 0.0
#                         else:
#                             autocorr_matrix = np.corrcoef(channel[:-1], channel[1:])
#                             autocorr = autocorr_matrix[0, 1]
                            
#                             if np.isnan(autocorr) or np.isinf(autocorr):
#                                 autocorr = 0.0
                                
#                         temp_features.append(autocorr)
#                     except Exception as e:
#                         temp_features.append(0.0)
#                 else:
#                     temp_features.append(0.0)
            
#             # Pad to ensure consistent length (8 features for 4 channels)
#             while len(temp_features) < 8:
#                 temp_features.append(0.0)
            
#             result = np.array(temp_features[:8], dtype=np.float32)
            
#             # Final safety check
#             result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
            
#             return result
            
#         except Exception as e:
#             print(f"⚠️ Error in temporal feature computation: {e}")
#             return np.zeros(8, dtype=np.float32)
    
#     def _align_word_embeddings(self):
#         """Get word embeddings aligned with EEG data"""
#         word_features = []
#         embedding_dim = 300  # Default embedding dimension
        
#         # Determine embedding dimension from first available embedding
#         if self.word_embeddings:
#             first_word = next(iter(self.word_embeddings))
#             embedding_dim = len(self.word_embeddings[first_word])
        
#         for i in range(min(len(self.words), len(self.eeg_data))):
#             word = self.words[i]
#             if word in self.word_embeddings:
#                 word_features.append(self.word_embeddings[word])
#             else:
#                 # Use zero vector for unknown words
#                 word_features.append(np.zeros(embedding_dim))
                
#         return np.array(word_features)
    
#     def _create_joint_space(self, word_features, eeg_features):
#         """Create joint embedding space using canonical correlation analysis"""
#         try:
#             from sklearn.cross_decomposition import CCA
#             from sklearn.preprocessing import StandardScaler
#             from sklearn.impute import SimpleImputer
            
#             # Ensure we have data
#             if len(word_features) == 0 or len(eeg_features) == 0:
#                 print("⚠️ No data for joint space creation")
#                 return self._create_default_joint_space()
            
#             print(f"🔧 Processing {len(word_features)} word features and {len(eeg_features)} EEG features")
            
#             # Step 1: Handle NaN and infinite values
#             word_features_clean = self._clean_features(word_features, "word")
#             eeg_features_clean = self._clean_features(eeg_features, "EEG")
            
#             # Step 2: Ensure same number of samples
#             min_samples = min(len(word_features_clean), len(eeg_features_clean))
#             word_features_clean = word_features_clean[:min_samples]
#             eeg_features_clean = eeg_features_clean[:min_samples]
            
#             # Step 3: Check for constant features and remove them
#             word_features_clean = self._remove_constant_features(word_features_clean, "word")
#             eeg_features_clean = self._remove_constant_features(eeg_features_clean, "EEG")
            
#             # Step 4: Standardize features
#             word_scaler = StandardScaler()
#             eeg_scaler = StandardScaler()
            
#             word_features_scaled = word_scaler.fit_transform(word_features_clean)
#             eeg_features_scaled = eeg_scaler.fit_transform(eeg_features_clean)
            
#             # Step 5: Final NaN check after scaling
#             if np.isnan(word_features_scaled).any() or np.isnan(eeg_features_scaled).any():
#                 print("⚠️ NaN values after scaling, using default joint space")
#                 return self._create_default_joint_space()
            
#             # Step 6: Use CCA with reduced components
#             n_components = min(5, word_features_scaled.shape[1], eeg_features_scaled.shape[1], 
#                              word_features_scaled.shape[0])
            
#             if n_components < 1:
#                 print("⚠️ Insufficient components for CCA")
#                 return self._create_default_joint_space()
            
#             print(f"🔬 Running CCA with {n_components} components")
#             cca = CCA(n_components=n_components, max_iter=100, tol=1e-06)
            
#             word_proj, eeg_proj = cca.fit_transform(word_features_scaled, eeg_features_scaled)
            
#             # Step 7: Final check for NaN in projections
#             if np.isnan(word_proj).any() or np.isnan(eeg_proj).any():
#                 print("⚠️ NaN in CCA projections, using PCA fallback")
#                 return self._create_pca_joint_space(word_features_clean, eeg_features_clean)
            
#             # Joint space combines both projections
#             joint_embeddings = np.concatenate([word_proj, eeg_proj], axis=1)
            
#             print(f"✅ Created joint space with shape {joint_embeddings.shape}")
            
#             return {
#                 'joint_embeddings': joint_embeddings,
#                 'word_projection': word_proj,
#                 'eeg_projection': eeg_proj,
#                 'cca_model': cca,
#                 'word_scaler': word_scaler,
#                 'eeg_scaler': eeg_scaler
#             }
            
#         except Exception as e:
#             print(f"⚠️ CCA failed with error: {e}")
#             print("🔧 Falling back to PCA-based joint space")
#             return self._create_pca_joint_space(word_features, eeg_features)
    
    
#     def _clean_features(self, features, feature_type):
#         """Clean features by handling NaN, inf, and extreme values"""
#         print(f"🧹 Cleaning {feature_type} features...")
        
#         # Convert to numpy array if not already
#         features_clean = np.array(features, dtype=np.float32)
        
#         # Handle NaN values
#         nan_mask = np.isnan(features_clean)
#         if nan_mask.any():
#             print(f"  Found {nan_mask.sum()} NaN values, replacing with zeros")
#             features_clean[nan_mask] = 0.0
        
#         # Handle infinite values
#         inf_mask = np.isinf(features_clean)
#         if inf_mask.any():
#             print(f"  Found {inf_mask.sum()} infinite values, clipping")
#             features_clean = np.clip(features_clean, -1e6, 1e6)
        
#         # Handle extreme values (beyond 5 standard deviations)
#         for col in range(features_clean.shape[1]):
#             col_data = features_clean[:, col]
#             if np.std(col_data) > 0:
#                 mean_val = np.mean(col_data)
#                 std_val = np.std(col_data)
#                 extreme_mask = np.abs(col_data - mean_val) > 5 * std_val
#                 if extreme_mask.any():
#                     features_clean[extreme_mask, col] = mean_val
        
#         print(f"  {feature_type} features cleaned: shape {features_clean.shape}")
#         return features_clean
    
#     def _remove_constant_features(self, features, feature_type):
#         """Remove features with zero variance"""
#         print(f"🔍 Checking for constant {feature_type} features...")
        
#         # Calculate variance for each feature
#         variances = np.var(features, axis=0)
#         non_constant_mask = variances > 1e-8  # Very small threshold
        
#         if not non_constant_mask.all():
#             n_removed = (~non_constant_mask).sum()
#             print(f"  Removed {n_removed} constant {feature_type} features")
#             features = features[:, non_constant_mask]
        
#         if features.shape[1] == 0:
#             print(f"⚠️ All {feature_type} features were constant, using random features")
#             features = np.random.randn(features.shape[0], 5).astype(np.float32)
        
#         return features
    
#     def _create_pca_joint_space(self, word_features, eeg_features):
#         """Create joint space using PCA as fallback"""
#         try:
#             from sklearn.decomposition import PCA
#             from sklearn.preprocessing import StandardScaler
            
#             print("🔧 Creating PCA-based joint space...")
            
#             # Clean features
#             word_features_clean = self._clean_features(word_features, "word")
#             eeg_features_clean = self._clean_features(eeg_features, "EEG")
            
#             # Ensure same number of samples
#             min_samples = min(len(word_features_clean), len(eeg_features_clean))
#             word_features_clean = word_features_clean[:min_samples]
#             eeg_features_clean = eeg_features_clean[:min_samples]
            
#             # Remove constant features
#             word_features_clean = self._remove_constant_features(word_features_clean, "word")
#             eeg_features_clean = self._remove_constant_features(eeg_features_clean, "EEG")
            
#             # Standardize
#             word_scaler = StandardScaler()
#             eeg_scaler = StandardScaler()
            
#             word_scaled = word_scaler.fit_transform(word_features_clean)
#             eeg_scaled = eeg_scaler.fit_transform(eeg_features_clean)
            
#             # Apply PCA to reduce dimensions
#             word_pca = PCA(n_components=min(10, word_scaled.shape[1], word_scaled.shape[0]))
#             eeg_pca = PCA(n_components=min(10, eeg_scaled.shape[1], eeg_scaled.shape[0]))
            
#             word_proj = word_pca.fit_transform(word_scaled)
#             eeg_proj = eeg_pca.fit_transform(eeg_scaled)
            
#             # Create joint embeddings
#             joint_embeddings = np.concatenate([word_proj, eeg_proj], axis=1)
            
#             print(f"✅ Created PCA joint space with shape {joint_embeddings.shape}")
            
#             return {
#                 'joint_embeddings': joint_embeddings,
#                 'word_projection': word_proj,
#                 'eeg_projection': eeg_proj,
#                 'cca_model': None,
#                 'word_pca': word_pca,
#                 'eeg_pca': eeg_pca,
#                 'word_scaler': word_scaler,
#                 'eeg_scaler': eeg_scaler
#             }
            
#         except Exception as e:
#             print(f"⚠️ PCA fallback failed: {e}")
#             return self._create_default_joint_space()
    
#     def _analyze_frequency_patterns(self):
#         """Analyze joint space to determine optimal frequency-specific filters"""
#         print("🎵 Analyzing frequency patterns for architecture design...")
        
#         try:
#             # Cluster joint embeddings
#             n_clusters = min(3, len(self.joint_space['joint_embeddings']))  # emotion, concrete, abstract
#             if n_clusters < 2:
#                 return self._create_default_frequency_mappings()
                
#             kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
#             clusters = kmeans.fit_predict(self.joint_space['joint_embeddings'])
            
#             frequency_mappings = {}
            
#             for cluster_id in range(n_clusters):
#                 cluster_mask = clusters == cluster_id
#                 cluster_indices = [i for i in range(len(clusters)) if cluster_mask[i]]
                
#                 if len(cluster_indices) == 0:
#                     continue
                    
#                 cluster_words = [self.words[i] for i in cluster_indices if i < len(self.words)]
                
#                 # Analyze EEG frequency patterns for this cluster
#                 cluster_eeg_features = [self.eeg_data[i] for i in cluster_indices if i < len(self.eeg_data)]
                
#                 if len(cluster_eeg_features) > 0:
#                     avg_eeg = np.mean(cluster_eeg_features, axis=0)
#                     freq_profile = self._compute_frequency_features(avg_eeg)
                    
#                     # Determine dominant frequencies
#                     dominant_freqs = self._find_dominant_frequencies(freq_profile)
                    
#                     # Generate filter specifications
#                     filter_specs = self._frequency_to_filters(dominant_freqs)
                    
#                     frequency_mappings[cluster_id] = {
#                         'words': cluster_words[:10],  # Sample words
#                         'frequency_profile': freq_profile,
#                         'dominant_frequencies': dominant_freqs,
#                         'filter_specs': filter_specs
#                     }
            
#             if not frequency_mappings:
#                 return self._create_default_frequency_mappings()
                
#             return frequency_mappings
            
#         except Exception as e:
#             print(f"⚠️ Frequency analysis failed: {e}")
#             return self._create_default_frequency_mappings()
    
#     def _create_default_frequency_mappings(self):
#         """Create default frequency mappings when analysis fails"""
#         return {
#             0: {
#                 'words': ['emotion', 'words'],
#                 'frequency_profile': np.ones(20),
#                 'dominant_frequencies': ['alpha', 'beta'],
#                 'filter_specs': [
#                     {'spatial': 2, 'temporal': 8, 'depth': 32, 'description': 'Balanced spatio-temporal for alpha'},
#                     {'spatial': 2, 'temporal': 4, 'depth': 48, 'description': 'Multi-channel for beta processing'}
#                 ]
#             },
#             1: {
#                 'words': ['noun', 'verb'],
#                 'frequency_profile': np.ones(20),
#                 'dominant_frequencies': ['theta', 'gamma'],
#                 'filter_specs': [
#                     {'spatial': 1, 'temporal': 12, 'depth': 24, 'description': 'Medium-wide temporal for theta'},
#                     {'spatial': 4, 'temporal': 2, 'depth': 64, 'description': 'Full spatial for gamma binding'}
#                 ]
#             }
#         }
    
#     def _find_dominant_frequencies(self, freq_profile):
#         """Identify which frequency bands are most important"""
#         # freq_profile has features for delta, theta, alpha, beta, gamma (4 channels each = 20 features)
#         band_powers = {
#             'delta': np.mean(freq_profile[0:4]) if len(freq_profile) >= 4 else 0,
#             'theta': np.mean(freq_profile[4:8]) if len(freq_profile) >= 8 else 0, 
#             'alpha': np.mean(freq_profile[8:12]) if len(freq_profile) >= 12 else 0,
#             'beta': np.mean(freq_profile[12:16]) if len(freq_profile) >= 16 else 0,
#             'gamma': np.mean(freq_profile[16:20]) if len(freq_profile) >= 20 else 0
#         }
        
#         # Find top 2 frequency bands
#         sorted_bands = sorted(band_powers.items(), key=lambda x: x[1], reverse=True)
#         dominant = [band[0] for band in sorted_bands[:2]]
        
#         return dominant
    
#     def _frequency_to_filters(self, dominant_frequencies):
#         """Convert dominant frequencies to optimal filter configurations"""
#         filter_configs = []
        
#         for freq_band in dominant_frequencies:
#             if freq_band == 'delta':
#                 filter_configs.append({
#                     'spatial': 1, 'temporal': 16, 'depth': 16,
#                     'description': 'Wide temporal for delta waves'
#                 })
#             elif freq_band == 'theta':
#                 filter_configs.append({
#                     'spatial': 1, 'temporal': 12, 'depth': 24,
#                     'description': 'Medium-wide temporal for theta'
#                 })
#             elif freq_band == 'alpha':
#                 filter_configs.append({
#                     'spatial': 2, 'temporal': 8, 'depth': 32,
#                     'description': 'Balanced spatio-temporal for alpha'
#                 })
#             elif freq_band == 'beta':
#                 filter_configs.append({
#                     'spatial': 2, 'temporal': 4, 'depth': 48,
#                     'description': 'Multi-channel for beta processing'
#                 })
#             elif freq_band == 'gamma':
#                 filter_configs.append({
#                     'spatial': 4, 'temporal': 2, 'depth': 64,
#                     'description': 'Full spatial for gamma binding'
#                 })
        
#         return filter_configs

# # -------------------------
# # ENHANCED MULTI-FREQUENCY BRANCHES - FIXED
# # -------------------------
# def create_enhanced_eeg_processor(input_shape, frequency_mappings=None):
#     """
#     Create enhanced EEG processor with frequency-specific branches
#     """
#     eeg_input = Input(shape=input_shape, name="eeg_input")  # Shape: (4, 60, 1)
    
#     if frequency_mappings:
#         print(f"🏗️ Creating {len(frequency_mappings)} frequency-specific branches...")
        
#         # Create branches based on embedding analysis
#         processed_outputs = []
        
#         for cluster_id, mapping in frequency_mappings.items():
#             branch_name = f"cluster_{cluster_id}"
#             x = eeg_input  # Start with the 4D input: (batch, 4, 60, 1)
            
#             print(f"  Building {branch_name} with {len(mapping['filter_specs'])} filters...")
            
#             # Apply each filter specification to the same input
#             for i, spec in enumerate(mapping['filter_specs']):
#                 # Ensure filter dimensions are valid for 4D input
#                 spatial_size = min(spec['spatial'], input_shape[0])  # Max 4 channels
#                 temporal_size = min(spec['temporal'], input_shape[1])  # Max 60 time points
                
#                 print(f"    Filter {i}: {spatial_size}x{temporal_size}, depth={spec['depth']}")
                
#                 x = Conv2D(
#                     filters=spec['depth'],
#                     kernel_size=(spatial_size, temporal_size),
#                     activation='relu',
#                     padding='same',
#                     name=f'{branch_name}_conv_{i}'
#                 )(x)
                
#                 # Only apply pooling if we have enough spatial/temporal dimensions
#                 pool_size = (1, min(2, x.shape[2]))  # Don't pool beyond available dimensions
#                 if x.shape[2] > 1:  # Only pool if temporal dimension > 1
#                     x = MaxPooling2D(pool_size, name=f'{branch_name}_pool_{i}')(x)
                
#                 x = BatchNormalization(name=f'{branch_name}_bn_{i}')(x)
#                 x = Dropout(0.2, name=f'{branch_name}_dropout_{i}')(x)
                
#                 print(f"    After filter {i}: {x.shape}")
            
#             # Global pooling for this branch
#             x = GlobalAveragePooling2D(name=f'{branch_name}_global_pool')(x)
#             processed_outputs.append(x)
            
#             print(f"  {branch_name} final output shape: {x.shape}")
        
#         # Combine all branch outputs
#         if len(processed_outputs) > 1:
#             print(f"🔗 Combining {len(processed_outputs)} branch outputs...")
#             combined = Concatenate(name='enhanced_fusion')(processed_outputs)
#         else:
#             combined = processed_outputs[0]
        
#         # Additional fusion layer
#         fused = Dense(128, activation='relu', name='branch_fusion')(combined)
#         print(f"🎯 Final enhanced features shape: {fused.shape}")
        
#     else:
#         print("🔄 Using fallback architecture...")
#         # Fallback to simple architecture
#         x = Conv2D(32, (2, 8), activation='relu', padding='same', name='fallback_conv1')(eeg_input)
#         x = MaxPooling2D((1, 2), name='fallback_pool1')(x)
#         x = Conv2D(64, (1, 4), activation='relu', padding='same', name='fallback_conv2')(x)
#         x = MaxPooling2D((1, 2), name='fallback_pool2')(x)
#         x = BatchNormalization(name='fallback_bn')(x)
#         x = GlobalAveragePooling2D(name='fallback_global_pool')(x)
#         fused = x
    
#     return eeg_input, fused

# # -------------------------
# # TRAINING UTILITIES
# # -------------------------
# def create_sample_data():
#     """Create sample data if real data files don't exist"""
#     print("📋 Creating sample data for training...")
    
#     # Create sample EEG data
#     n_samples_emotion = 1000
#     n_samples_pos = 800
    
#     X_emotion = np.random.randn(n_samples_emotion, 4, 60).astype(np.float32)
#     y_emotion = np.random.randint(0, 3, n_samples_emotion)
#     words_emotion = [f"emotion_word_{i}" for i in range(n_samples_emotion)]
    
#     X_pos = np.random.randn(n_samples_pos, 4, 60).astype(np.float32)
#     y_pos = np.random.randint(0, 2, n_samples_pos)  # Already 0-1 for POS
#     words_pos = [f"pos_word_{i}" for i in range(n_samples_pos)]
    
#     emotion_data = {"data": X_emotion, "labels": y_emotion, "words": words_emotion}
#     pos_data = {"data": X_pos, "labels": y_pos, "words": words_pos}
    
#     return emotion_data, pos_data

# def create_sample_embeddings():
#     """Create sample embeddings if file doesn't exist"""
#     print("📋 Creating sample embeddings...")
    
#     embedding_dict = {}
#     for i in range(2000):
#         word = f"word_{i}"
#         embedding = np.random.randn(300).astype(np.float32)
#         embedding_dict[word] = embedding
    
#     # Add some specific words
#     for word in ['emotion', 'happy', 'sad', 'angry', 'noun', 'verb', 'action', 'object']:
#         embedding_dict[word] = np.random.randn(300).astype(np.float32)
    
#     return embedding_dict

# # -------------------------
# # ENHANCED MAIN EXECUTION - TRAINABLE
# # -------------------------
# def main():
#     print("🚀 Enhanced Strategy: Multi-Modal Embedding + Multi-Frequency Architecture - TRAINABLE")
    
#     # Load or create embeddings
#     try:
#         embedding_dict = load_embeddings(EMBEDDINGS_PATH)
#     except:
#         print("📋 Using sample embeddings...")
#         embedding_dict = create_sample_embeddings()
    
#     # Load or create datasets
#     try:
#         print("📂 Loading datasets...")
#         emotion_data = load_pickle(EMOTION_DATA_PATH)
#         pos_data = load_pickle(POS_DATA_PATH)
#     except:
#         print("📋 Using sample data...")
#         emotion_data, pos_data = create_sample_data()
    
#     # Prepare datasets
#     X_e = np.array(emotion_data["data"]).reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
#     y_e = np.array(emotion_data["labels"])
#     words_e = emotion_data.get("words", [f"emotion_word_{i}" for i in range(len(X_e))])
    
#     X_p = np.array(pos_data["data"]).reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
#     y_p = np.array(pos_data["labels"])
#     # Ensure POS labels are 0-1
#     if y_p.max() > 1:
#         y_p = y_p - 1
#     words_p = pos_data.get("words", [f"pos_word_{i}" for i in range(len(X_p))])
    
#     print(f"📊 Data shapes: Emotion {X_e.shape}, POS {X_p.shape}")
#     print(f"📊 Label ranges: Emotion {y_e.min()}-{y_e.max()}, POS {y_p.min()}-{y_p.max()}")
    
#     # Combine data for embedding analysis
#     X_combined_raw = np.concatenate([X_e, X_p], axis=0)
#     all_words = list(words_e) + list(words_p)
#     all_labels = list(y_e) + list(y_p)
    
#     # ENHANCED: Create multi-modal embedding analyzer
#     frequency_mappings = None
#     if embedding_dict:
#         print("🔬 Creating multi-modal embedding space...")
#         analyzer = MultiModalEmbeddingAnalyzer(
#             word_embeddings=embedding_dict,
#             eeg_data=X_combined_raw,
#             words=all_words,
#             labels=all_labels
#         )
        
#         # Create joint space and analyze frequency patterns
#         joint_space = analyzer.create_joint_embedding_space()
#         frequency_mappings = analyzer.frequency_mappings
        
#         # Print discovered patterns
#         print("🎯 Discovered frequency patterns:")
#         for cluster_id, mapping in frequency_mappings.items():
#             print(f"  Cluster {cluster_id}: {mapping['words'][:3]}...")
#             print(f"    Dominant frequencies: {mapping['dominant_frequencies']}")
#             for spec in mapping['filter_specs']:
#                 print(f"    Filter: {spec['description']} ({spec['spatial']}, {spec['temporal']}, {spec['depth']})")
    
#     # Preprocessing
#     print("🔧 Preprocessing data...")
#     X_e_norm = robust_normalize_eeg(X_e)
#     X_p_norm = robust_normalize_eeg(X_p)
    
#     X_e_norm = X_e_norm[..., np.newaxis]
#     X_p_norm = X_p_norm[..., np.newaxis]
    
#     X_combined = np.concatenate([X_e_norm, X_p_norm], axis=0)
    
#     # Create labels with proper shapes
#     y_emotion_combined = np.concatenate([
#         to_categorical(y_e, NUM_CLASSES_EMOTION),
#         np.zeros((len(X_p_norm), NUM_CLASSES_EMOTION))
#     ], axis=0)
    
#     y_pos_combined = np.concatenate([
#         np.zeros((len(X_e_norm), NUM_CLASSES_POS)),
#         to_categorical(y_p, NUM_CLASSES_POS)
#     ], axis=0)
    
#     # Sample weights for multi-task learning
#     sw_emotion = np.concatenate([np.ones(len(X_e_norm)), np.zeros(len(X_p_norm))], axis=0)
#     sw_pos = np.concatenate([np.zeros(len(X_e_norm)), np.ones(len(X_p_norm))], axis=0)
    
#     print("🏗️ Building enhanced multi-frequency architecture...")
    
#     # ENHANCED: Build model with embedding-driven architecture
#     eeg_input, eeg_features = create_enhanced_eeg_processor(INPUT_SHAPE, frequency_mappings)
    
#     print(f"📊 EEG features shape: {eeg_features.shape}")
    
#     # Shared dense layers
#     x = Dense(FUSION_DENSE, activation="relu", name="shared_dense1")(eeg_features)
#     x = BatchNormalization(name="shared_bn1")(x)
#     x = Dropout(0.3, name="shared_dropout1")(x)
#     x = Dense(FUSION_DENSE//2, activation="relu", name="shared_dense2")(x)
#     x = BatchNormalization(name="shared_bn2")(x)
#     x = Dropout(0.2, name="shared_dropout2")(x)
    
#     # Task-specific heads
#     emotion_logits = Dense(NUM_CLASSES_EMOTION, activation='softmax', name='emotion_output')(x)
#     pos_logits = Dense(NUM_CLASSES_POS, activation='softmax', name='pos_output')(x)
    
#     # Build enhanced model
#     model = Model(inputs=eeg_input, outputs=[emotion_logits, pos_logits], 
#                   name="enhanced_embedding_architecture")
    
#     # Compile model
#     model.compile(
#         optimizer=Adam(learning_rate=LR),
#         loss={
#             'emotion_output': 'categorical_crossentropy',
#             'pos_output': 'categorical_crossentropy'
#         },
#         loss_weights={'emotion_output': 1.0, 'pos_output': 1.0},
#         metrics=[['accuracy'], ["accuracy"]]
#     )
    
#     print("📊 Enhanced model summary:")
#     model.summary()
    
#     # Check data integrity before training
#     check_data_integrity(X_combined, y_emotion_combined, "Combined Emotion")
#     check_data_integrity(X_combined, y_pos_combined, "Combined POS")
    
#     # Train-test split
#     print("🔀 Splitting data for training...")
    
#     # Create stratification labels based on sample weights
#     # Use emotion labels where sw_emotion > 0, POS labels where sw_pos > 0
#     stratify_labels = []
#     for i in range(len(sw_emotion)):
#         if sw_emotion[i] > 0:  # This is an emotion sample
#             emotion_class = np.argmax(y_emotion_combined[i])
#             stratify_labels.append(f"emotion_{emotion_class}")
#         else:  # This is a POS sample
#             pos_class = np.argmax(y_pos_combined[i])
#             stratify_labels.append(f"pos_{pos_class}")
    
#     # Convert to numeric labels for stratification
#     from sklearn.preprocessing import LabelEncoder
#     label_encoder = LabelEncoder()
#     stratify_numeric = label_encoder.fit_transform(stratify_labels)
    
#     X_train, X_test, y_e_train, y_e_test, y_p_train, y_p_test = train_test_split(
#         X_combined, y_emotion_combined, y_pos_combined, 
#         test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_numeric
#     )
    
#     # Split sample weights too
#     sw_e_train, sw_e_test = train_test_split(sw_emotion, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_numeric)
#     sw_p_train, sw_p_test = train_test_split(sw_pos, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify_numeric)
    
#     print(f"📊 Training data shape: {X_train.shape}")
#     print(f"📊 Test data shape: {X_test.shape}")
    
#     # Callbacks
#     callbacks = [
#         EarlyStopping(
#             monitor='val_loss',
#             patience=15,
#             restore_best_weights=True,
#             verbose=1
#         ),
#         ReduceLROnPlateau(
#             monitor='val_loss',
#             factor=0.5,
#             patience=8,
#             min_lr=1e-7,
#             verbose=1
#         ),
#         ModelCheckpoint(
#             'enhanced_embedding_model_best.h5',
#             monitor='val_loss',
#             save_best_only=True,
#             verbose=1
#         )
#     ]
    
#     # FIXED: Train the model with proper data handling
#     print("🎯 Starting training...")
#     print("=" * 60)
    
#     # Create separate training runs for each task to avoid sample weight issues
#     print("🔄 Training emotion task...")
#     emotion_mask_train = sw_e_train > 0
#     emotion_mask_test = sw_e_test > 0
    
#     if np.sum(emotion_mask_train) > 0 and np.sum(emotion_mask_test) > 0:
#         X_emotion_train = X_train[emotion_mask_train]
#         y_emotion_train = y_e_train[emotion_mask_train]
#         X_emotion_test = X_test[emotion_mask_test]
#         y_emotion_test = y_e_test[emotion_mask_test]
        
#         # Create single-task emotion model for initial training
#         emotion_model = Model(inputs=eeg_input, outputs=emotion_logits)
#         emotion_model.compile(
#             optimizer=Adam(learning_rate=LR),
#             loss='categorical_crossentropy',
#             metrics=['accuracy']
#         )
        
#         print(f"Training emotion model with {len(X_emotion_train)} samples...")
#         emotion_history = emotion_model.fit(
#             X_emotion_train, y_emotion_train,
#             validation_data=(X_emotion_test, y_emotion_test),
#             batch_size=BATCH_SIZE,
#             epochs=min(20, EPOCHS),
#             callbacks=callbacks[:2],  # Skip ModelCheckpoint for subtask
#             verbose=1
#         )
    
#     print("🔄 Training POS task...")
#     pos_mask_train = sw_p_train > 0
#     pos_mask_test = sw_p_test > 0
    
#     if np.sum(pos_mask_train) > 0 and np.sum(pos_mask_test) > 0:
#         X_pos_train = X_train[pos_mask_train]
#         y_pos_train = y_p_train[pos_mask_train]
#         X_pos_test = X_test[pos_mask_test]
#         y_pos_test = y_p_test[pos_mask_test]
        
#         # Create single-task POS model
#         pos_model = Model(inputs=eeg_input, outputs=pos_logits)
#         pos_model.compile(
#             optimizer=Adam(learning_rate=LR),
#             loss='categorical_crossentropy',
#             metrics=['accuracy']
#         )
        
#         print(f"Training POS model with {len(X_pos_train)} samples...")
#         pos_history = pos_model.fit(
#             X_pos_train, y_pos_train,
#             validation_data=(X_pos_test, y_pos_test),
#             batch_size=BATCH_SIZE,
#             epochs=min(20, EPOCHS),
#             callbacks=callbacks[:2],  # Skip ModelCheckpoint for subtask
#             verbose=1
#         )
    
#     # Now train the full multi-task model with balanced sampling
#     print("🎯 Fine-tuning multi-task model...")
    
#     # Create balanced batches by sampling equal amounts from each task
#     n_emotion_train = np.sum(emotion_mask_train)
#     n_pos_train = np.sum(pos_mask_train)
    
#     if n_emotion_train > 0 and n_pos_train > 0:
#         # Sample equal amounts from each task
#         n_per_task = min(n_emotion_train, n_pos_train)
        
#         emotion_indices = np.where(emotion_mask_train)[0][:n_per_task]
#         pos_indices = np.where(pos_mask_train)[0][:n_per_task]
        
#         # Create balanced training set
#         balanced_indices = np.concatenate([emotion_indices, pos_indices])
#         np.random.shuffle(balanced_indices)
        
#         X_balanced = X_train[balanced_indices]
#         y_e_balanced = y_e_train[balanced_indices]
#         y_p_balanced = y_p_train[balanced_indices]
#         sw_e_balanced = sw_e_train[balanced_indices]
#         sw_p_balanced = sw_p_train[balanced_indices]
        
#         # Train multi-task model with balanced data
#         history = model.fit(
#             X_balanced,
#             [y_e_balanced, y_p_balanced],  # Use list instead of dictionary
#             validation_data=(
#                 X_test,
#                 [y_e_test, y_p_test]  # Use list instead of dictionary
#             ),
#             batch_size=BATCH_SIZE,
#             epochs=min(30, EPOCHS),
#             callbacks=callbacks,
#             verbose=1
#         )
#     else:
#         print("⚠️ Insufficient data for multi-task training, using single-task results")
#         # Create dummy history
#         history = type('DummyHistory', (), {'history': {'loss': [0.5], 'val_loss': [0.6]}})()
    
#     # Evaluate the model
#     print("\n" + "=" * 60)
#     print("📈 Evaluating model...")
    
#     # Evaluate on separate tasks
#     if np.sum(emotion_mask_test) > 0:
#         emotion_pred = model.predict(X_test[emotion_mask_test])[0]
#         emotion_true = y_e_test[emotion_mask_test]
        
#         emotion_pred_classes = np.argmax(emotion_pred, axis=1)
#         emotion_true_classes = np.argmax(emotion_true, axis=1)
        
#         emotion_accuracy = np.mean(emotion_pred_classes == emotion_true_classes)
#         print(f"📊 Emotion Test Accuracy: {emotion_accuracy:.4f}")
        
#         print("\n🎭 Emotion Classification Report:")
#         print(classification_report(
#             emotion_true_classes,
#             emotion_pred_classes,
#             target_names=['Class_0', 'Class_1', 'Class_2']
#         ))
    
#     if np.sum(pos_mask_test) > 0:
#         pos_pred = model.predict(X_test[pos_mask_test])[1]
#         pos_true = y_p_test[pos_mask_test]
        
#         pos_pred_classes = np.argmax(pos_pred, axis=1)
#         pos_true_classes = np.argmax(pos_true, axis=1)
        
#         pos_accuracy = np.mean(pos_pred_classes == pos_true_classes)
#         print(f"📊 POS Test Accuracy: {pos_accuracy:.4f}")
        
#         print("\n📝 POS Classification Report:")
#         print(classification_report(
#             pos_true_classes,
#             pos_pred_classes,
#             target_names=['Noun', 'Verb']
#         ))
    
#     # Plot training history if available
#     if hasattr(history, 'history') and history.history:
#         print("📈 Plotting training history...")
#         plot_training_history(history)
    
#     # Save final model
#     model_save_path = f"enhanced_embedding_model_final_{np.random.randint(1000, 9999)}.h5"
#     model.save(model_save_path)
#     print(f"💾 Model saved as: {model_save_path}")
    
#     print("\n✅ Enhanced Training Complete!")
#     print("🧠 Architecture automatically designed from joint word-EEG embedding space!")
#     print("🎵 Multi-frequency filters optimized for discovered neural patterns!")
    
#     return model, history

# def plot_training_history(history):
#     """Plot training history"""
#     try:
#         fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
#         # Total loss
#         axes[0, 0].plot(history.history.get('loss', []), label='Training Loss')
#         axes[0, 0].plot(history.history.get('val_loss', []), label='Validation Loss')
#         axes[0, 0].set_title('Total Loss')
#         axes[0, 0].set_xlabel('Epoch')
#         axes[0, 0].set_ylabel('Loss')
#         axes[0, 0].legend()
#         axes[0, 0].grid(True)
        
#         # Emotion accuracy
#         axes[0, 1].plot(history.history.get('emotion_output_accuracy', []), label='Training Emotion Acc')
#         axes[0, 1].plot(history.history.get('val_emotion_output_accuracy', []), label='Validation Emotion Acc')
#         axes[0, 1].set_title('Emotion Classification Accuracy')
#         axes[0, 1].set_xlabel('Epoch')
#         axes[0, 1].set_ylabel('Accuracy')
#         axes[0, 1].legend()
#         axes[0, 1].grid(True)
        
#         # POS accuracy
#         axes[1, 0].plot(history.history.get('pos_output_accuracy', []), label='Training POS Acc')
#         axes[1, 0].plot(history.history.get('val_pos_output_accuracy', []), label='Validation POS Acc')
#         axes[1, 0].set_title('POS Classification Accuracy')
#         axes[1, 0].set_xlabel('Epoch')
#         axes[1, 0].set_ylabel('Accuracy')
#         axes[1, 0].legend()
#         axes[1, 0].grid(True)
        
#         # Learning rate (if available)
#         if 'lr' in history.history:
#             axes[1, 1].plot(history.history['lr'])
#             axes[1, 1].set_title('Learning Rate')
#             axes[1, 1].set_xlabel('Epoch')
#             axes[1, 1].set_ylabel('Learning Rate')
#             axes[1, 1].set_yscale('log')
#             axes[1, 1].grid(True)
#         else:
#             axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
#                            ha='center', va='center', transform=axes[1, 1].transAxes)
        
#         plt.tight_layout()
#         plt.savefig('enhanced_training_history.png', dpi=300, bbox_inches='tight')
#         plt.show()
#         print("📊 Training history plot saved as 'enhanced_training_history.png'")
        
#     except Exception as e:
#         print(f"⚠️ Could not plot training history: {e}")

# # -------------------------
# # HELPER FUNCTIONS
# # -------------------------
# def load_pickle(path):
#     """Load pickle file with error handling"""
#     try:
#         with open(path, "rb") as f:
#             return pickle.load(f)
#     except FileNotFoundError:
#         raise FileNotFoundError(f"File {path} not found")
#     except Exception as e:
#         raise Exception(f"Error loading {path}: {e}")

# def load_embeddings(path):
#     """Load embeddings with error handling"""
#     if path is None or not os.path.exists(path):
#         raise FileNotFoundError(f"Embeddings path {path} not found")
    
#     data = load_pickle(path)
    
#     # Handle different embedding formats
#     if isinstance(data, dict):
#         return {w: np.array(vec, dtype=np.float32) for w, vec in data.items()}
#     else:
#         raise ValueError("Embeddings file format not recognized")

# def robust_normalize_eeg(X):
#     """Standard robust normalization with error handling"""
#     try:
#         Xn = X.copy().astype(np.float32)
        
#         for ch in range(Xn.shape[1]):
#             channel_data = Xn[:, ch, :]
            
#             # Use percentile-based normalization
#             p5 = np.percentile(channel_data, 5, axis=1, keepdims=True)
#             p95 = np.percentile(channel_data, 95, axis=1, keepdims=True)
#             scale = p95 - p5
            
#             # Avoid division by zero
#             scale = np.where(scale == 0, 1, scale)
            
#             # Normalize
#             Xn[:, ch, :] = (channel_data - p5) / scale
        
#         return Xn
    
#     except Exception as e:
#         print(f"⚠️ Error in normalization: {e}")
#         # Fallback to simple standardization
#         return (X - np.mean(X, axis=(1, 2), keepdims=True)) / (np.std(X, axis=(1, 2), keepdims=True) + 1e-8)

# def check_data_integrity(X, y, name):
#     """Check data integrity and print statistics"""
#     print(f"\n📊 {name} Data Integrity Check:")
#     print(f"  Shape: {X.shape}")
#     print(f"  Labels shape: {y.shape}")
#     print(f"  Data range: [{X.min():.4f}, {X.max():.4f}]")
#     print(f"  Labels range: [{y.min()}, {y.max()}]")
#     print(f"  NaN values: {np.isnan(X).sum()}")
#     print(f"  Inf values: {np.isinf(X).sum()}")
    
#     if len(y.shape) > 1:
#         print(f"  Label distribution: {[np.sum(y[:, i]) for i in range(y.shape[1])]}")
#     else:
#         unique, counts = np.unique(y, return_counts=True)
#         print(f"  Label distribution: {dict(zip(unique, counts))}")

# # -------------------------
# # MAIN EXECUTION
# # -------------------------
# if __name__ == "__main__":
#     # Set up GPU if available
#     try:
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             print(f"🎮 Found {len(gpus)} GPU(s)")
#             for gpu in gpus:
#                 tf.config.experimental.set_memory_growth(gpu, True)
#         else:
#             print("💻 Using CPU")
#     except Exception as e:
#         print(f"⚠️ GPU setup warning: {e}")
    
#     # Run main training
#     try:
#         model, history = main()
#         print("\n🎉 Training completed successfully!")
        
#     except KeyboardInterrupt:
#         print("\n⚠️ Training interrupted by user")
        
#     except Exception as e:
#         print(f"\n❌ Training failed with error: {e}")
#         import traceback
#         traceback.print_exc()
        
#         # Provide fallback simple model for testing
#         print("\n🔧 Creating simple fallback model for testing...")
#         try:
#             simple_input = Input(shape=INPUT_SHAPE, name="simple_input")
#             x = Conv2D(32, (2, 8), activation='relu', padding='same')(simple_input)
#             x = MaxPooling2D((1, 2))(x)
#             x = Conv2D(64, (1, 4), activation='relu', padding='same')(x)
#             x = GlobalAveragePooling2D()(x)
#             x = Dense(128, activation='relu')(x)
#             x = Dropout(0.3)(x)
            
#             emotion_out = Dense(NUM_CLASSES_EMOTION, activation='softmax', name='emotion_output')(x)
#             pos_out = Dense(NUM_CLASSES_POS, activation='softmax', name='pos_output')(x)
            
#             fallback_model = Model(inputs=simple_input, outputs=[emotion_out, pos_out])
#             fallback_model.compile(
#                 optimizer=Adam(learning_rate=1e-4),
#                 loss=['categorical_crossentropy', 'categorical_crossentropy'],
#                 metrics=[['accuracy'], ["accuracy"]]
#             )
            
#             print("✅ Fallback model created successfully!")
#             fallback_model.summary()
            
#         except Exception as fallback_error:
#             print(f"❌ Even fallback model failed: {fallback_error}")