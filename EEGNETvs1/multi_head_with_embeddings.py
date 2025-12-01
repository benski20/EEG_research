# improved_multi_task_fusion_balanced_neurosymbolic_with_contrastive.py
# FULL: Balanced fusion + FFT-aware bandpower head + word-embedding anchoring + InfoNCE contrastive loss
from tensorflow.keras.layers import Layer, Lambda, Activation
import os
import pickle
import hashlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten, Lambda, Add, Multiply
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

# -------------------------
# CONFIG (tune these)
# -------------------------
EMOTION_MODEL_PATH = "eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5"
POS_MODEL_PATH = "eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5"
EMOTION_DATA_PATH = "eeg_fft_dataset.pkl"
POS_DATA_PATH = "eeg_fft_nounvsverb_JASON_dataset.pkl"

# Optional resources (set paths if you have them)
WORDLIST_EMOTION_PATH = "emotion_words.pkl"  # e.g. "emotion_words.pkl"
WORDLIST_POS_PATH = "pos_upd_words.pkl"     # e.g. "pos_words.pkl"
EMBEDDINGS_PATH = "embeddings.pkl"        # e.g. "embeddings.pkl" where embeddings is dict word->vec

INPUT_SHAPE = (4, 60, 1)   # (channels, freq_bins, channel-last)
NUM_CLASSES_EMOTION = 3
NUM_CLASSES_POS = 2
FUSION_DENSE = 256
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-5
RANDOM_STATE = 42

# Embedding & bandpower hyperparams
EMB_DIM = 64
LAMBDA_EMB = 0.025       # cosine embedding loss weight
LAMBDA_BP = 0.08        # bandpower MSE loss weight
LAMBDA_CONTRAST = 0.02  # InfoNCE contrastive loss weight (via layer.add_loss)
TAU = 0.1               # temperature for InfoNCE
K_GATE = 0.01           # gating slope for gating between EEG and embedding logits
FREQ_MAX_HZ = 60.0      # upper frequency for FFT bins (adjust if needed)

# -------------------------
# GENTLE FOCAL LOSS - BALANCED APPROACH
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

# -------------------------
# InfoNCE loss layer (adds loss to model)
# -------------------------
class InfoNCELossLayer(tf.keras.layers.Layer):
    """
    Computes InfoNCE (contrastive) loss between predicted EEG embeddings and true word embeddings.
    Adds self.weight * loss to model via add_loss.
    Inputs: [emb_pred, emb_true] where both are shape (batch, emb_dim)
    """
    def __init__(self, weight=0.1, tau=0.1, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight
        self.tau = tau

    def call(self, inputs, training=None):
        emb_pred, emb_true = inputs  # tensors
        # normalize
        emb_pred_n = tf.math.l2_normalize(emb_pred, axis=1)
        emb_true_n = tf.math.l2_normalize(emb_true, axis=1)

        # similarity matrix S_ij = emb_pred_i · emb_true_j / tau
        logits = tf.matmul(emb_pred_n, emb_true_n, transpose_b=True) / self.tau  # (B,B)
        # labels: positive is diagonal
        batch_size = tf.shape(logits)[0]
        labels = tf.range(batch_size)
        # cross entropy where each row i has logits[i] and label i is positive
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        self.add_loss(self.weight * loss)
        # Return emb_pred unchanged so layer can be placed in graph
        return emb_pred

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"weight": self.weight, "tau": self.tau})
        return cfg
from tensorflow.keras.layers import Layer
class L2Normalize(Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)
# -------------------------
# HELPER / PREPROCESSING
# -------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
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
    # if len(neutral_augmented) > 0:
    #     augmented_samples.append(np.array(neutral_augmented))
    #     augmented_labels.append(np.full(len(neutral_augmented), neutral_class))
    
    X_final = np.concatenate(augmented_samples, axis=0)
    y_final = np.concatenate(augmented_labels, axis=0)
    
    print(f"🎭 After conservative augmentation: {np.bincount(y_final)}")
    
    return X_final, y_final

def get_feature_extractor(model):
    """Extract features before final classification layer"""
    flatten_layer = None
    for layer in model.layers:
        if "flatten" in layer.name.lower():
            flatten_layer = layer
            break
    if flatten_layer is not None:
        return Model(inputs=model.input, outputs=flatten_layer.output)
    else:
        if len(model.layers) >= 2:
            return Model(inputs=model.input, outputs=model.layers[-2].output)
        else:
            raise ValueError("Could not identify a feature layer in model.")

def robust_normalize_eeg(X):
    """Standard robust normalization across freq bins per channel"""
    Xn = X.copy().astype(np.float32)
    for ch in range(Xn.shape[1]):
        channel_data = Xn[:, ch, :]
        p5 = np.percentile(channel_data, 5, axis=1, keepdims=True)
        p95 = np.percentile(channel_data, 95, axis=1, keepdims=True)
        scale = p95 - p5
        scale = np.where(scale == 0, 1, scale)
        Xn[:, ch, :] = (channel_data - p5) / scale
    return Xn

def create_stratified_split(X, y_emotion_1d, y_pos_1d, test_size=0.2, random_state=42):
    """Create stratified split considering both tasks (best-effort)"""
    combined_labels = []
    for i in range(len(X)):
        if i < len(y_emotion_1d):
            combined_labels.append(f"emotion_{y_emotion_1d[i]}")
        else:
            combined_labels.append(f"pos_{y_pos_1d[i - len(y_emotion_1d)]}")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, val_idx = next(sss.split(X, combined_labels))
    return train_idx, val_idx

# -------------------------
# FFT-AWARE BANDPOWER (for FFT-formatted data)
# -------------------------
def compute_bandpower_from_fft(X_fft, freq_max_hz=FREQ_MAX_HZ, bands=None):
    """
    X_fft: (n_samples, n_channels, n_freq_bins) containing power per freq bin.
    Assumes bins span 0..freq_max_hz uniformly.
    Returns: (n_samples, n_channels * n_bands) log-bandpower vector and band names.
    """
    if bands is None:
        bands = {'delta': (1,4), 'theta': (4,8), 'alpha': (8,12), 'beta': (12,30), 'lowgamma': (30,45)}
    n, ch, nbins = X_fft.shape
    freqs = np.linspace(0.0, freq_max_hz, nbins)
    bp = np.zeros((n, ch, len(bands)), dtype=np.float32)
    band_items = list(bands.items())
    for bi, (_name, (low, high)) in enumerate(band_items):
        idx = (freqs >= low) & (freqs <= high)
        if not np.any(idx):
            continue
        bp[:, :, bi] = np.sum(X_fft[:, :, idx], axis=2)
    bp = np.maximum(bp, 1e-9)
    bp = np.log(bp)
    return bp.reshape(n, -1), [k for k,_ in band_items]

# -------------------------
# WORD EMBEDDINGS UTILITIES
# -------------------------
def deterministic_word_embedding(word, emb_dim=EMB_DIM, seed=42):
    """
    Deterministic pseudo-embedding for fallback (hash -> RNG).
    Returns normalized vector.
    """
    h = int(hashlib.md5(word.encode('utf-8')).hexdigest()[:8], 16)
    rng = np.random.RandomState(seed + (h % 10000))
    v = rng.normal(size=(emb_dim,))
    v = v / (np.linalg.norm(v) + 1e-9)
    return v.astype(np.float32)

def load_embeddings(path):
    """
    loads dict: word -> vector
    If path None or missing, returns None.
    """
    if path is None:
        return None
    if not os.path.exists(path):
        print(f"⚠️ Embeddings path {path} not found — falling back to deterministic embeddings.")
        return None
    data = load_pickle(path)
    return {w: np.array(vec, dtype=np.float32) for w, vec in data.items()}
import keras
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

# -------------------------
# SIMPLE ATTENTION FUSION (unchanged)
# -------------------------
def create_simple_attention_fusion(emotion_features, pos_features, fusion_dim=256):
    emotion_proj = Dense(fusion_dim//2, activation='relu', name='emotion_proj')(emotion_features)
    pos_proj = Dense(fusion_dim//2, activation='relu', name='pos_proj')(pos_features)
    # simple attention
    emotion_attention = Dense(1, activation='sigmoid', name='emotion_att')(emotion_proj)
    pos_attention = Dense(1, activation='sigmoid', name='pos_att')(pos_proj)
    emotion_weighted = Lambda(lambda x: x[0] * x[1])([emotion_proj, emotion_attention])
    pos_weighted = Lambda(lambda x: x[0] * x[1])([pos_proj, pos_attention])
    fused = Concatenate()([emotion_weighted, pos_weighted])
    return fused
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
# COSINE LOSS (for embedding alignment)
# -------------------------
def cosine_loss(y_true, y_pred):
    y_true = tf.math.l2_normalize(y_true, axis=1)
    y_pred = tf.math.l2_normalize(y_pred, axis=1)
    return 1.0 - tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=1))

# -------------------------
# MAIN EXECUTION
# -------------------------
from keras import config
import tensorflow as tf
from tensorflow.keras.layers import Layer

def main():
    print("🚀 Loading pretrained models (feature extractors)...")
    emotion_model = load_model(EMOTION_MODEL_PATH)
    pos_model = load_model(POS_MODEL_PATH)

    emotion_extractor = get_feature_extractor(emotion_model)
    pos_extractor = get_feature_extractor(pos_model)
    emotion_extractor.trainable = False
    pos_extractor.trainable = False

    # Build model
    print("🏗️ Building neurosymbolic-enhanced fusion model with contrastive loss...")
    main_input = Input(shape=INPUT_SHAPE, name="eeg_input")  # (4,60,1)
    feat_a_tensor = emotion_extractor(main_input)
    feat_b_tensor = pos_extractor(main_input)
    feat_a_flat = Flatten(name="emotion_features_flat")(feat_a_tensor)
    feat_b_flat = Flatten(name="pos_features_flat")(feat_b_tensor)
    fused_features = create_simple_attention_fusion(feat_a_flat, feat_b_flat, fusion_dim=FUSION_DENSE)

    x = Dense(FUSION_DENSE, activation="relu", name="fusion_dense1")(fused_features)
    x = BatchNormalization(name="fusion_bn1")(x)
    x = Dropout(0.3, name="fusion_dropout1")(x)
    x = Dense(FUSION_DENSE//2, activation="relu", name="fusion_dense2")(x)
    x = BatchNormalization(name="fusion_bn2")(x)
    x = Dropout(0.2, name="fusion_dropout2")(x)

    # EEG -> logits heads
    logits_eeg_em = Dense(NUM_CLASSES_EMOTION, name='logits_eeg_em')(x)
    logits_eeg_pos = Dense(NUM_CLASSES_POS, name='logits_eeg_pos')(x)

    # EEG -> predicted embedding
    

    # ---- Custom layers ----
    
    @keras.saving.register_keras_serializable()
    class L2Normalize(Layer):
        def call(self, inputs):
            return tf.math.l2_normalize(inputs, axis=1)

    @keras.saving.register_keras_serializable()
    class CosineSimilarity(Layer):
        def call(self, inputs):
            x1, x2 = inputs
            return tf.reduce_sum(x1 * x2, axis=1, keepdims=True)

    @keras.saving.register_keras_serializable()
    class SigmoidGating(Layer):
        def __init__(self, k_gate=1.0, **kwargs):
            super().__init__(**kwargs)
            self.k_gate = k_gate

        def call(self, inputs):
            return tf.sigmoid(self.k_gate * inputs)

    @keras.saving.register_keras_serializable()
    class OneMinus(Layer):
        def call(self, inputs):
            return 1.0 - inputs

    @keras.saving.register_keras_serializable() 
    class SoftmaxLayer(Layer):
        def __init__(self, axis=-1, **kwargs):
            super().__init__(**kwargs)
            self.axis = axis

        def call(self, inputs):
            return tf.nn.softmax(inputs, axis=self.axis)


    @keras.saving.register_keras_serializable()
    class BalancedGatingLayer(Layer):
        """
        Custom layer for balanced gating between EEG and embedding logits
        """
        def __init__(self, k_gate=1.0, min_eeg_weight=0.975, **kwargs):
            super().__init__(**kwargs)
            self.k_gate = k_gate
            self.min_eeg_weight = min_eeg_weight
            
        def call(self, inputs):
            cos_sim = inputs
            
            # Apply sigmoid with gating strength
            gating_raw = tf.sigmoid(self.k_gate * cos_sim)
            
            # Ensure EEG always has minimum influence
            balanced_gate = gating_raw * (1 - self.min_eeg_weight)
            eeg_weight = 1.0 - balanced_gate
            
            return balanced_gate, eeg_weight
        
        def get_config(self):
            config = super().get_config()
            config.update({
                'k_gate': self.k_gate,
                'min_eeg_weight': self.min_eeg_weight
            })
            return config
    # ---- Model block ----

    emb_pred = Dense(EMB_DIM, activation=None, name='emb_pred')(x)
    emb_pred_norm = L2Normalize(name="emb_pred_norm")(emb_pred)

    # Word embedding (true) input
    word_emb_input = Input(shape=(EMB_DIM,), name='word_emb_input')
    word_emb_norm = L2Normalize(name="word_emb_norm")(word_emb_input)

    # Map word embedding -> logits (semantic prior)
    emb_to_em = Dense(128, activation='relu', name='emb_to_em_dense')(word_emb_input)
    logits_emb_em = Dense(NUM_CLASSES_EMOTION, name='logits_emb_em')(emb_to_em)
    emb_to_pos = Dense(64, activation='relu', name='emb_to_pos_dense')(word_emb_input)
    logits_emb_pos = Dense(NUM_CLASSES_POS, name='logits_emb_pos')(emb_to_pos)

    # gating by cosine similarity emb_pred <-> word_emb
    # Progressive or balanced gating
    cos_sim = CosineSimilarity(name='cos_sim')([emb_pred_norm, word_emb_norm])

    # Option 1: Balanced gating (immediate fix)
    # Create the balanced gating layer
    balanced_gating_layer = BalancedGatingLayer(
        k_gate=1.0,           # Much gentler than your original 6.0
        min_eeg_weight=0.9,   # Ensures at least 30% EEG influence
        name='balanced_gating'
    )
    
    # Apply the layer - it returns TWO outputs
    gating, one_minus_gating = balanced_gating_layer(cos_sim)


    # combine logits
    logits_combined_em = Add(name='logits_combined_em')([
        Multiply()([logits_eeg_em, one_minus_gating]),
        Multiply()([logits_emb_em, gating])
    ])
    logits_combined_pos = Add(name='logits_combined_pos')([
        Multiply()([logits_eeg_pos, one_minus_gating]),
        Multiply()([logits_emb_pos, gating])
    ])

    # Emotion output softmax
    emotion_out = SoftmaxLayer(axis=-1, name='emotion')(logits_combined_em)
    pos_out = SoftmaxLayer(axis=-1, name='pos')(logits_combined_pos)


    # Bandpower head placeholder: real size will be attached after computing band dims
    # For now we construct band_pred later and rebuild final model.

    # Add InfoNCE layer: it will add its loss to the model graph when connected
    info_nce_layer = InfoNCELossLayer(weight=LAMBDA_CONTRAST, tau=TAU, name='info_nce')
    # connect layer (passes emb_pred through, adds loss internally)
    _ = info_nce_layer([emb_pred, word_emb_input])

    # Build partial model now; we will attach band_pred (size-known) and rebuild full model
    partial_model = Model(inputs=[main_input, word_emb_input], outputs=[emotion_out, pos_out, emb_pred, emb_pred_norm], name='partial_neurosym')

    print("📂 Loading datasets...")
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data = load_pickle(POS_DATA_PATH)

    # Prepare emotion dataset
    X_e = np.array(emotion_data["data"]).reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
    y_e = np.array(emotion_data["labels"])
    words_e = None
    if "words" in emotion_data:
        words_e = list(emotion_data["words"])
    elif WORDLIST_EMOTION_PATH is not None and os.path.exists(WORDLIST_EMOTION_PATH):
        words_e = load_pickle(WORDLIST_EMOTION_PATH)

    # Preprocess + conservative improvements
    # gentle_emotion_preprocessing expects X_emotion as (n,ch,freq)
    X_e_proc = gentle_emotion_preprocessing(X_e)
    X_e_aug, y_e_aug, emotion_class_weights = apply_conservative_emotion_improvements(X_e_proc, y_e)

    # Prepare POS dataset
    X_p = np.array(pos_data["data"]).reshape(-1, INPUT_SHAPE[0], INPUT_SHAPE[1])
    y_p = np.array(pos_data["labels"]) - 1
    words_p = None
    if "words" in pos_data:
        words_p = list(pos_data["words"])
    elif WORDLIST_POS_PATH is not None and os.path.exists(WORDLIST_POS_PATH):
        words_p = load_pickle(WORDLIST_POS_PATH)

    X_p_proc = robust_normalize_eeg(X_p)

    # Light SMOTE for POS if needed
    X_p_balanced, y_p_balanced = X_p_proc, y_p
    pos_counts = np.bincount(y_p)
    if len(pos_counts) > 1 and np.max(pos_counts) > np.min(pos_counts) * 1.5:
        print("⚖️ Applying SMOTE to POS data...")
        original_shape = X_p_proc.shape
        X_p_flat = X_p_proc.reshape(X_p_proc.shape[0], -1)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_p_flat, y_p_balanced = smote.fit_resample(X_p_flat, y_p)
        X_p_balanced = X_p_flat.reshape(-1, *original_shape[1:])

    print(f"📈 Data shapes (before channel dim add): Emotion: {X_e.shape} -> {X_e_aug.shape}, POS: {X_p.shape} -> {X_p_balanced.shape}")

    # Add channel-last dim
    X_e_aug = X_e_aug[..., np.newaxis]
    X_p_balanced = X_p_balanced[..., np.newaxis]

    # Channel weights multiplication
    channel_weights = np.array([1.0, 1.0, 0.75, 0.75]).reshape(1, 4, 1, 1)
    X_e_aug = X_e_aug * channel_weights

    # Combine datasets
    X_combined = np.concatenate([X_e_aug, X_p_balanced], axis=0)

    # Categorical labels one-hot with zeros for missing head
    y_emotion_combined = np.concatenate([
        to_categorical(y_e_aug, NUM_CLASSES_EMOTION),
        np.zeros((len(X_p_balanced), NUM_CLASSES_EMOTION))
    ], axis=0)
    y_pos_combined = np.concatenate([
        np.zeros((len(X_e_aug), NUM_CLASSES_POS)),
        to_categorical(y_p_balanced, NUM_CLASSES_POS)
    ], axis=0)

    # sample weight masks
    sw_emotion = np.concatenate([np.ones(len(X_e_aug)), np.zeros(len(X_p_balanced))], axis=0)
    sw_pos = np.concatenate([np.zeros(len(X_e_aug)), np.ones(len(X_p_balanced))], axis=0)

    # Compute bandpower targets from FFT data (remove channel-last dim)
    X_combined_nodim = X_combined[..., 0]  # shape (N, ch, freq)
    band_true, band_names = compute_bandpower_from_fft(X_combined_nodim, freq_max_hz=FREQ_MAX_HZ)
    num_band_features = band_true.shape[1]
    print(f"🔊 Computed band features: {len(band_names)} bands -> band feature dims per sample: {num_band_features}")

    # Prepare word embeddings for each sample
    embedding_dict = load_embeddings(EMBEDDINGS_PATH)
    words_combined = []
    if words_e is not None:
        words_combined.extend(words_e if isinstance(words_e, list) else list(words_e))
    else:
        words_combined.extend([f"emotion_word_{i}" for i in range(len(X_e_aug))])
    if words_p is not None:
        words_combined.extend(words_p if isinstance(words_p, list) else list(words_p))
    else:
        words_combined.extend([f"pos_word_{i}" for i in range(len(X_p_balanced))])

    emb_true_mat = np.zeros((len(words_combined), EMB_DIM), dtype=np.float32)
    for i, w in enumerate(words_combined):
        if embedding_dict is not None and w in embedding_dict:
            vec = embedding_dict[w]
            if vec.shape[0] != EMB_DIM:
                v = np.zeros((EMB_DIM,), dtype=np.float32)
                l = min(len(v), len(vec))
                v[:l] = vec[:l]
                vec = v
            emb_true_mat[i] = vec / (np.linalg.norm(vec) + 1e-9)
        else:
            emb_true_mat[i] = deterministic_word_embedding(w, emb_dim=EMB_DIM, seed=RANDOM_STATE)

    # Attach band_pred Dense now that num_band_features known
    band_pred = Dense(num_band_features, activation=None, name='band_pred')(x)  # regression output

    # Build final model with band_pred added
    final_model = Model(inputs=[main_input, word_emb_input],
                        outputs=[emotion_out, pos_out, emb_pred, band_pred],
                        name="neurosymbolic_balanced_fusion_with_contrast")

    # Compile with multiple losses; InfoNCE was added via the InfoNCELossLayer earlier (it uses add_loss)
    print("⚙️ Compiling model with multi-loss (classification + emb + band power + contrastive)...")
    final_model.compile(
        optimizer=Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss=[
            balanced_focal_loss(alpha=0.25, gamma=1.5),  # emotion
            balanced_focal_loss(alpha=0.25, gamma=1.5),  # pos
            cosine_loss,                                  # emb_pred
            'mse'                                         # band_pred
        ],
        loss_weights=[1.2, 1.0, LAMBDA_EMB, LAMBDA_BP],
        metrics=[['accuracy'], ['accuracy'], [], []]
    )

    
    # Train/val split
    print("🎯 Creating stratified train/validation split...")
    train_idx, val_idx = create_stratified_split(X_combined, y_e_aug, y_p_balanced, test_size=0.2, random_state=RANDOM_STATE)
    X_train, X_val = X_combined[train_idx], X_combined[val_idx]
    ye_train, ye_val = y_emotion_combined[train_idx], y_emotion_combined[val_idx]
    yp_train, yp_val = y_pos_combined[train_idx], y_pos_combined[val_idx]
    emb_train, emb_val = emb_true_mat[train_idx], emb_true_mat[val_idx]
    band_train, band_val = band_true[train_idx], band_true[val_idx]
    swe_train, swe_val = sw_emotion[train_idx], sw_emotion[val_idx]
    swp_train, swp_val = sw_pos[train_idx], sw_pos[val_idx]

    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath='embedding_ns_fusion_model_true_tf.keras', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Fit
    print("🏋️ Starting training (this includes InfoNCE via add_loss)...")
    
    # Prepare input dictionaries
    x_train_dict = [X_train, emb_train]  # Use list format instead of dict
    x_val_dict = [X_val, emb_val]
    
    # Prepare output lists (matching model output order)
    y_train_list = [ye_train, yp_train, emb_train, band_train]
    y_val_list = [ye_val, yp_val, emb_val, band_val]
    
    # Prepare sample weights as list (matching output order)
    sample_weights_train = [swe_train, swp_train, np.ones(len(X_train)), np.ones(len(X_train))]
    sample_weights_val = [swe_val, swp_val, np.ones(len(X_val)), np.ones(len(X_val))]
    
    history = final_model.fit(
        x=x_train_dict,
        y=y_train_list,
        sample_weight=sample_weights_train,
        validation_data=(x_val_dict, y_val_list, sample_weights_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    config.enable_unsafe_deserialization()
    # Evaluation & save
    print("📊 Evaluation...")
    best_model = load_model('embedding_ns_fusion_model_true_tf.keras', custom_objects={
        'focal_loss_fixed': balanced_focal_loss(),
        'cosine_loss': cosine_loss,
        'InfoNCELossLayer': InfoNCELossLayer
    })
    # Create random embeddings for testing
    random_emb_val = np.random.normal(0, 1, emb_val.shape)
    random_emb_val = random_emb_val / (np.linalg.norm(random_emb_val, axis=1, keepdims=True) + 1e-9)

    y_pred = best_model.predict([X_val, random_emb_val])
    # outputs: [emotion_out, pos_out, emb_pred, band_pred]
    y_pred_emotion = y_pred[0]
    y_pred_pos = y_pred[1]
    y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
    y_pred_pos_classes = np.argmax(y_pred_pos, axis=1)
    ye_val_labels = np.argmax(ye_val, axis=1)
    yp_val_labels = np.argmax(yp_val, axis=1)

    # Emotion evaluation
    emotion_mask = swe_val > 0
    if np.sum(emotion_mask) > 0:
        print("\n🎭 BALANCED EMOTION CLASSIFICATION RESULTS:")
        print(classification_report(ye_val_labels[emotion_mask], y_pred_emotion_classes[emotion_mask], target_names=["Negative", "Neutral", "Positive"]))
        emotion_cm = confusion_matrix(ye_val_labels[emotion_mask], y_pred_emotion_classes[emotion_mask])
        print("Emotion Confusion Matrix:")
        print(emotion_cm)
        emotion_per_class_acc = np.diag(emotion_cm) / np.sum(emotion_cm, axis=1) * 100
        print(f"Per-class accuracy: Negative: {emotion_per_class_acc[0]:.1f}%, Neutral: {emotion_per_class_acc[1]:.1f}%, Positive: {emotion_per_class_acc[2]:.1f}%")

    # POS evaluation
    pos_mask = swp_val > 0
    if np.sum(pos_mask) > 0:
        print("\n📝 POS CLASSIFICATION RESULTS:")
        print(classification_report(yp_val_labels[pos_mask], y_pred_pos_classes[pos_mask], target_names=["Noun", "Verb"]))

    # Save final model
    best_model.save("best_balanced_fusion_model_neurosymbolic_contrastive.h5")
    print("\n✅ Completed. Best model saved as best_balanced_fusion_model_neurosymbolic_contrastive.h5")

    return best_model

if __name__ == "__main__":
    main()
