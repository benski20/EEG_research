# neurosymbolic_balanced_fusion.py
# — Balanced multi-task fusion with neurosymbolic regularization (Keras 3-safe)

import os
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
POS_MODEL_PATH     = "eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5"
EMOTION_DATA_PATH  = "eeg_fft_dataset.pkl"
POS_DATA_PATH      = "eeg_fft_nounvsverb_JASON_dataset.pkl"
VALENCE_LEXICON_PATH = "valence_lexicon.pkl"  # optional; dict: word -> [p_neg, p_neu, p_pos]

INPUT_SHAPE = (4, 60, 1)
NUM_CLASSES_EMOTION = 3  # [Negative, Neutral, Positive]
NUM_CLASSES_POS     = 2  # [Noun, Verb]
NUM_PREDICATES      = 5  # latent EEG predicates (FA_L, FA_R, OccAlphaDrop, MotorBetaL, MotorBetaR)

FUSION_DENSE = 256
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
RANDOM_STATE = 42

# -------------------------
# NEUROSYMBOLIC HYPERPARAMETERS
# -------------------------
LMB_SEMANTIC   = 0.20   # lexicon prior KL (gated by confidence)
LMB_LOGIC      = 0.50   # fuzzy rule penalties (predicates -> valence)
LMB_COMPAT     = 0.10   # PoS↔Valence compatibility prior
LMB_PRED_REG   = 0.02   # predicate mean activation regularizer
CONF_TAU       = 0.65   # confidence threshold for lexicon gating
PRED_TARGET_MEAN = 0.20 # target mean predicate activation

# -------------------------
# LOSS
# -------------------------
def balanced_focal_loss(alpha=0.25, gamma=1.5):
    def focal_loss_fixed(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -y_true * tf.math.log(y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = alpha_t * tf.pow((1 - p_t), gamma)
        focal_loss = focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    return focal_loss_fixed

# -------------------------
# HELPERS
# -------------------------
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
        # fall back to penultimate
        if len(model.layers) >= 2:
            feature_output = model.layers[-2].output
        else:
            raise ValueError("Could not identify a feature layer in model.")
    return Model(inputs=model.input, outputs=feature_output)

def robust_normalize_eeg(X):
    X = X.copy()
    for ch in range(X.shape[1]):
        channel_data = X[:, ch, :]
        p5  = np.percentile(channel_data, 5,  axis=1, keepdims=True)
        p95 = np.percentile(channel_data, 95, axis=1, keepdims=True)
        scale = p95 - p5
        scale = np.where(scale == 0, 1, scale)
        X[:, ch, :] = (channel_data - p5) / scale
    return X

def gentle_emotion_preprocessing(X_emotion):
    Xp = X_emotion.copy().astype(np.float32)
    for ch in range(Xp.shape[1]):
        channel_data = Xp[:, ch, :]
        p25 = np.percentile(channel_data, 25, axis=1, keepdims=True)
        p75 = np.percentile(channel_data, 75, axis=1, keepdims=True)
        scale = p75 - p25
        scale = np.where(scale == 0, 1, scale)
        Xp[:, ch, :] = (channel_data - p25) / scale
    return Xp

def conservative_neutral_augmentation(X_emotion, y_emotion, neutral_class=1):
    neutral_mask = (y_emotion == neutral_class)
    neutral_samples = X_emotion[neutral_mask]
    print(f"🎭 Original neutral samples: {np.sum(neutral_mask)}")
    if len(neutral_samples) == 0:
        return X_emotion, y_emotion
    augmented_samples = [X_emotion]
    augmented_labels  = [y_emotion]
    neutral_augmented = []
    for sample in neutral_samples:
        aug = sample.copy()
        noise_std = 0.01 * np.std(aug)
        noise = np.random.normal(0, noise_std, aug.shape)
        aug += noise
        neutral_augmented.append(aug)
    if len(neutral_augmented) > 0:
        augmented_samples.append(np.array(neutral_augmented))
        augmented_labels.append(np.full(len(neutral_augmented), neutral_class))
    X_final = np.concatenate(augmented_samples, axis=0)
    y_final = np.concatenate(augmented_labels, axis=0)
    print(f"🎭 After conservative augmentation: {np.bincount(y_final)}")
    return X_final, y_final

def conservative_class_weights(y_emotion):
    classes = np.unique(y_emotion)
    standard = compute_class_weight('balanced', classes=classes, y=y_emotion)
    weights = dict(zip(classes, standard))
    if 1 in weights:
        weights[1] *= 1.3
    print(f"🎯 Conservative class weights: {weights}")
    return weights

def apply_conservative_emotion_improvements(X_emotion, y_emotion):
    print("🎯 Applying conservative emotion improvements...")
    print("🔧 Light preprocessing...")
    Xp = gentle_emotion_preprocessing(X_emotion)

    print("🎭 Conservative neutral augmentation...")
    Xa, ya = conservative_neutral_augmentation(Xp, y_emotion)

    print("⚖️ Conservative SMOTE...")
    original_shape = Xa.shape
    X_flat = Xa.reshape(Xa.shape[0], -1)
    counts = np.bincount(ya)
    if counts.max() > counts.min() * 1.5:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_flat, ya = smote.fit_resample(X_flat, ya)
        Xb = X_flat.reshape(-1, *original_shape[1:])
    else:
        print("📊 Classes reasonably balanced, skipping SMOTE")
        Xb = Xa

    class_weights = conservative_class_weights(ya)
    print(f"📊 Final emotion dataset shape: {Xb.shape}")
    print(f"📊 Final emotion distribution: {np.bincount(ya)}")
    return Xb, ya, class_weights

def create_simple_attention_fusion(emotion_features, pos_features, fusion_dim=256):
    emotion_proj = Dense(fusion_dim//2, activation='relu', name='emotion_proj')(emotion_features)
    pos_proj     = Dense(fusion_dim//2, activation='relu', name='pos_proj')(pos_features)

    emotion_att = Dense(1, activation='sigmoid', name='emotion_att')(emotion_proj)
    pos_att     = Dense(1, activation='sigmoid', name='pos_att')(pos_proj)

    emotion_weighted = Lambda(lambda x: x[0] * x[1])([emotion_proj, emotion_att])
    pos_weighted     = Lambda(lambda x: x[0] * x[1])([pos_proj, pos_att])
    return Concatenate()([emotion_weighted, pos_weighted])

def maybe_load_valence_lexicon(path):
    if os.path.exists(path):
        try:
            d = load_pickle(path)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return None

def build_lexicon_priors_for_emotion_samples(words, lexicon):
    """
    words: list[str] or None. Returns np.ndarray (N,3).
    If word not found or words is None, returns zeros row (signals 'no prior').
    """
    N = len(words) if words is not None else 0
    if N == 0 or lexicon is None:
        return None
    priors = []
    for w in words:
        if w in lexicon:
            p = np.array(lexicon[w], dtype=np.float32)
            p = p / (p.sum() + 1e-8)
            priors.append(p)
        else:
            priors.append(np.zeros((NUM_CLASSES_EMOTION,), dtype=np.float32))
    return np.stack(priors, axis=0)

# ============================================================
# 🔹 Logic Constraint Layer (Keras 3–safe; metrics tracked via tf.keras.metrics.Mean)
# ============================================================
class LogicConstraintLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_val=NUM_CLASSES_EMOTION,
        num_pos=NUM_CLASSES_POS,
        lmb_sem=LMB_SEMANTIC,
        lmb_logic=LMB_LOGIC,
        lmb_compat=LMB_COMPAT,
        lmb_pred_reg=LMB_PRED_REG,
        conf_tau=CONF_TAU,
        pred_target_mean=PRED_TARGET_MEAN,
        name="logic_constraints",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.num_val = num_val
        self.num_pos = num_pos
        self.lmb_sem = lmb_sem
        self.lmb_logic = lmb_logic
        self.lmb_compat = lmb_compat
        self.lmb_pred_reg = lmb_pred_reg
        self.conf_tau = conf_tau
        self.pred_target_mean = pred_target_mean

        # Learned PoS↔Val compatibility logits (2×3)
        self.compat_logits = self.add_weight(
            name="compat_logits",
            shape=(self.num_pos * self.num_val,),
            initializer="zeros",
            trainable=True,
        )

        # Keras 3-safe tracked metrics
        self.m_sem   = tf.keras.metrics.Mean(name="loss_semantic")
        self.m_logic = tf.keras.metrics.Mean(name="loss_logic")
        self.m_comp  = tf.keras.metrics.Mean(name="loss_compat")
        self.m_pred  = tf.keras.metrics.Mean(name="loss_pred_reg")

    @staticmethod
    def _kl(p, q, eps=1e-6):
        p = tf.clip_by_value(p, eps, 1.0)
        q = tf.clip_by_value(q, eps, 1.0)
        return tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(q)), axis=-1)

    def call(self, inputs):
        """
        inputs = [emotion_probs, pos_probs, predicate_probs, lex_prior]
        - emotion_probs: (B, 3) softmax
        - pos_probs:     (B, 2) softmax
        - predicate_probs: (B, P) sigmoid
        - lex_prior:     (B, 3) valence prior or zeros if unknown
        """
        e_probs, p_probs, pred_probs, lex_prior = inputs
        eps = 1e-6

        # -------- Semantic (lexicon) loss with confidence gating --------
        has_prior = tf.cast(tf.reduce_sum(lex_prior, axis=-1, keepdims=True) > 0.0, tf.float32)
        conf = tf.reduce_max(e_probs, axis=-1, keepdims=True)
        gate = tf.cast(conf < self.conf_tau, tf.float32) * has_prior
        kl_sem = self._kl(e_probs, lex_prior)
        loss_semantic = tf.reduce_mean(kl_sem * tf.squeeze(gate, axis=-1)) * self.lmb_sem

        # -------- Fuzzy logic rules (Łukasiewicz implication) -----------
        # Example rules using the first two predicates as frontal asymmetry proxies
        fa_left  = pred_probs[:, 0]   # ⇒ Positive tendency
        fa_right = pred_probs[:, 1]   # ⇒ Negative tendency
        val_neg = e_probs[:, 0]
        val_pos = e_probs[:, 2]
        truth_r1 = tf.minimum(1.0, 1.0 - fa_left  + val_pos)
        truth_r2 = tf.minimum(1.0, 1.0 - fa_right + val_neg)
        viol_r1  = 1.0 - truth_r1
        viol_r2  = 1.0 - truth_r2
        loss_logic = tf.reduce_mean(0.5 * (viol_r1 + viol_r2)) * self.lmb_logic

        # -------- PoS↔Valence compatibility KL -------------------------
        p_joint = tf.einsum('bi,bj->bij', p_probs, e_probs)  # (B,2,3)
        compat_probs = tf.nn.softmax(self.compat_logits)     # (6,)
        compat_probs = tf.reshape(compat_probs, (self.num_pos, self.num_val))  # (2,3)
        kl_joint = tf.reduce_sum(
            p_joint * (
                tf.math.log(tf.clip_by_value(p_joint, eps, 1.0)) -
                tf.math.log(tf.clip_by_value(compat_probs, eps, 1.0))
            ),
            axis=[1, 2]
        )
        loss_compat = tf.reduce_mean(kl_joint) * self.lmb_compat

        # -------- Predicate regularization ------------------------------
        mean_pred = tf.reduce_mean(pred_probs, axis=-1)  # (B,)
        loss_pred_reg = tf.reduce_mean(tf.square(mean_pred - self.pred_target_mean)) * self.lmb_pred_reg

        total = loss_semantic + loss_logic + loss_compat + loss_pred_reg

        # Track metrics (Keras will surface these)
        self.m_sem.update_state(loss_semantic)
        self.m_logic.update_state(loss_logic)
        self.m_comp.update_state(loss_compat)
        self.m_pred.update_state(loss_pred_reg)

        # Attach total loss to graph
        self.add_loss(total)

        # Return a dummy tensor (not used further)
        batch = tf.shape(e_probs)[0]
        return e_probs

    @property
    def metrics(self):
        # Expose metric objects so they appear in model.history
        return [self.m_sem, self.m_logic, self.m_comp, self.m_pred]

    def compute_output_shape(self, input_shape):
        # (batch, 1)
        return (input_shape[0][0], 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(
            num_val=self.num_val,
            num_pos=self.num_pos,
            lmb_sem=self.lmb_sem,
            lmb_logic=self.lmb_logic,
            lmb_compat=self.lmb_compat,
            lmb_pred_reg=self.lmb_pred_reg,
            conf_tau=self.conf_tau,
            pred_target_mean=self.pred_target_mean,
        ))
        return cfg

# -------------------------
# MAIN
# -------------------------
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

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add this function after your other helper functions
def plot_confusion_matrices(y_true_emotion, y_pred_emotion, y_true_pos, y_pred_pos, 
                          emotion_mask, pos_mask, save_path=None):
    """
    Plot confusion matrices for both emotion and POS classification tasks
    
    Args:
        y_true_emotion: True emotion labels (full validation set)
        y_pred_emotion: Predicted emotion labels (full validation set) 
        y_true_pos: True POS labels (full validation set)
        y_pred_pos: Predicted POS labels (full validation set)
        emotion_mask: Boolean mask for emotion samples
        pos_mask: Boolean mask for POS samples
        save_path: Optional path to save the plots
    """
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Confusion Matrices - Neurosymbolic Multi-Task EEG Classification', 
                 fontsize=16, fontweight='bold')
    
    # Define class names
    emotion_classes = ['Negative', 'Neutral', 'Positive']
    pos_classes = ['Noun', 'Verb']
    
    # Plot 1: Emotion Classification Confusion Matrix
    if np.sum(emotion_mask) > 0:
        emotion_cm = confusion_matrix(y_true_emotion[emotion_mask], 
                                    y_pred_emotion[emotion_mask])
        
        # Calculate percentages
        emotion_cm_pct = emotion_cm.astype('float') / emotion_cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both counts and percentages
        annotations = []
        for i in range(emotion_cm.shape[0]):
            row = []
            for j in range(emotion_cm.shape[1]):
                count = emotion_cm[i, j]
                pct = emotion_cm_pct[i, j]
                row.append(f'{count}\n({pct:.1f}%)')
            annotations.append(row)
        
        # Plot emotion confusion matrix
        sns.heatmap(emotion_cm_pct, 
                   annot=annotations, 
                   fmt='',
                   cmap='Blues',
                   xticklabels=emotion_classes,
                   yticklabels=emotion_classes,
                   ax=axes[0],
                   cbar_kws={'label': 'Percentage (%)'})
        
        axes[0].set_title('Emotion Classification\n(Valence Detection)', 
                         fontsize=14, fontweight='bold', pad=20)
        axes[0].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('True Class', fontsize=12, fontweight='bold')
        
        # Add sample count and accuracy info
        total_emotion_samples = np.sum(emotion_mask)
        emotion_accuracy = np.sum(y_true_emotion[emotion_mask] == y_pred_emotion[emotion_mask]) / total_emotion_samples * 100
        axes[0].text(0.02, 0.98, f'N = {total_emotion_samples}\nAcc = {emotion_accuracy:.1f}%', 
                    transform=axes[0].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add per-class accuracy
        per_class_acc = np.diag(emotion_cm) / np.sum(emotion_cm, axis=1) * 100
        per_class_text = '\n'.join([f'{cls}: {acc:.1f}%' for cls, acc in zip(emotion_classes, per_class_acc)])
        axes[0].text(0.98, 0.02, per_class_text, 
                    transform=axes[0].transAxes, fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    else:
        axes[0].text(0.5, 0.5, 'No Emotion Data Available', 
                    transform=axes[0].transAxes, ha='center', va='center', fontsize=14)
        axes[0].set_title('Emotion Classification', fontsize=14, fontweight='bold')
    
    # Plot 2: POS Classification Confusion Matrix  
    if np.sum(pos_mask) > 0:
        pos_cm = confusion_matrix(y_true_pos[pos_mask], 
                                y_pred_pos[pos_mask])
        
        # Calculate percentages
        pos_cm_pct = pos_cm.astype('float') / pos_cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create annotations with both counts and percentages
        annotations = []
        for i in range(pos_cm.shape[0]):
            row = []
            for j in range(pos_cm.shape[1]):
                count = pos_cm[i, j]
                pct = pos_cm_pct[i, j] 
                row.append(f'{count}\n({pct:.1f}%)')
            annotations.append(row)
        
        # Plot POS confusion matrix
        sns.heatmap(pos_cm_pct,
                   annot=annotations,
                   fmt='',
                   cmap='Greens',
                   xticklabels=pos_classes,
                   yticklabels=pos_classes,
                   ax=axes[1],
                   cbar_kws={'label': 'Percentage (%)'})
        
        axes[1].set_title('Part-of-Speech Classification\n(Noun vs Verb)', 
                         fontsize=14, fontweight='bold', pad=20)
        axes[1].set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('True Class', fontsize=12, fontweight='bold')
        
        # Add sample count and accuracy info
        total_pos_samples = np.sum(pos_mask)
        pos_accuracy = np.sum(y_true_pos[pos_mask] == y_pred_pos[pos_mask]) / total_pos_samples * 100
        axes[1].text(0.02, 0.98, f'N = {total_pos_samples}\nAcc = {pos_accuracy:.1f}%', 
                    transform=axes[1].transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Add per-class accuracy
        per_class_acc = np.diag(pos_cm) / np.sum(pos_cm, axis=1) * 100
        per_class_text = '\n'.join([f'{cls}: {acc:.1f}%' for cls, acc in zip(pos_classes, per_class_acc)])
        axes[1].text(0.98, 0.02, per_class_text, 
                    transform=axes[1].transAxes, fontsize=9, verticalalignment='bottom',
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    else:
        axes[1].text(0.5, 0.5, 'No POS Data Available', 
                    transform=axes[1].transAxes, ha='center', va='center', fontsize=14)
        axes[1].set_title('POS Classification', fontsize=14, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Add timestamp and model info at bottom
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fig.text(0.5, 0.02, f'Generated: {timestamp} | Neurosymbolic Multi-Task EEG Classification', 
             ha='center', fontsize=10, style='italic', color='gray')
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Confusion matrices saved to: {save_path}")
    
    # Show plot
    plt.show()
    
    return fig

def plot_training_history(history, save_path=None):
    """
    Plot training history including losses and neurosymbolic components
    
    Args:
        history: Training history object from model.fit()
        save_path: Optional path to save the plot
    """
    
    # Extract history data
    epochs = range(1, len(history.history['loss']) + 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History - Neurosymbolic Multi-Task EEG Model', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Overall Loss
    axes[0, 0].plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    if 'val_loss' in history.history:
        axes[0, 0].plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Overall Loss', fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Task-specific Losses
    emotion_loss_key = 'emotion_loss'
    pos_loss_key = 'pos_loss'
    
    # Try to find the actual keys (they might have different names)
    for key in history.history.keys():
        if 'emotion' in key.lower() and 'loss' in key.lower() and 'val' not in key.lower():
            emotion_loss_key = key
        elif 'pos' in key.lower() and 'loss' in key.lower() and 'val' not in key.lower():
            pos_loss_key = key
    
    if emotion_loss_key in history.history:
        axes[0, 1].plot(epochs, history.history[emotion_loss_key], 'b-', 
                       label='Emotion Loss', linewidth=2)
    if pos_loss_key in history.history:
        axes[0, 1].plot(epochs, history.history[pos_loss_key], 'g-', 
                       label='POS Loss', linewidth=2)
    axes[0, 1].set_title('Task-Specific Losses', fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Accuracies
    emotion_acc_key = 'emotion_accuracy'
    pos_acc_key = 'pos_accuracy'
    
    for key in history.history.keys():
        if 'emotion' in key.lower() and 'acc' in key.lower() and 'val' not in key.lower():
            emotion_acc_key = key
        elif 'pos' in key.lower() and 'acc' in key.lower() and 'val' not in key.lower():
            pos_acc_key = key
    
    if emotion_acc_key in history.history:
        axes[0, 2].plot(epochs, history.history[emotion_acc_key], 'b-', 
                       label='Emotion Acc', linewidth=2)
    if pos_acc_key in history.history:
        axes[0, 2].plot(epochs, history.history[pos_acc_key], 'g-', 
                       label='POS Acc', linewidth=2)
    axes[0, 2].set_title('Task Accuracies', fontweight='bold')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Neurosymbolic Loss Components
    ns_components = ['loss_semantic', 'loss_logic', 'loss_compat', 'loss_pred_reg']
    colors = ['purple', 'orange', 'brown', 'pink']
    
    for i, (component, color) in enumerate(zip(ns_components, colors)):
        if component in history.history:
            axes[1, 0].plot(epochs, history.history[component], color=color, 
                           label=component.replace('loss_', ''), linewidth=2)
    
    axes[1, 0].set_title('Neurosymbolic Loss Components', fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')  # Log scale for better visibility
    
    # Plot 5: Learning Rate (if available)
    if 'lr' in history.history:
        axes[1, 1].plot(epochs, history.history['lr'], 'r-', linewidth=2)
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                       transform=axes[1, 1].transAxes, ha='center', va='center')
        axes[1, 1].set_title('Learning Rate Schedule', fontweight='bold')
    
    # Plot 6: Validation Metrics Summary
    val_metrics = []
    val_values = []
    
    for key in history.history.keys():
        if 'val_' in key and key != 'val_loss':
            val_metrics.append(key.replace('val_', ''))
            val_values.append(history.history[key][-1])  # Final epoch value
    
    if val_metrics:
        bars = axes[1, 2].bar(range(len(val_metrics)), val_values, 
                             color=['lightblue', 'lightgreen', 'lightyellow', 'lightcoral'][:len(val_metrics)])
        axes[1, 2].set_title('Final Validation Metrics', fontweight='bold')
        axes[1, 2].set_xlabel('Metrics')
        axes[1, 2].set_ylabel('Value')
        axes[1, 2].set_xticks(range(len(val_metrics)))
        axes[1, 2].set_xticklabels(val_metrics, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, val_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    else:
        axes[1, 2].text(0.5, 0.5, 'Validation Metrics\nNot Available', 
                       transform=axes[1, 2].transAxes, ha='center', va='center')
        axes[1, 2].set_title('Final Validation Metrics', fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"📊 Training history saved to: {save_path}")
    
    plt.show()
    
    return fig

def main():
    print("🚀 Loading pretrained models...")
    emotion_model = load_model(EMOTION_MODEL_PATH)
    pos_model     = load_model(POS_MODEL_PATH)

    print("🔧 Creating feature extractors...")
    emotion_extractor = get_feature_extractor(emotion_model)
    pos_extractor     = get_feature_extractor(pos_model)

    emotion_extractor.trainable = False
    pos_extractor.trainable     = False

    dummy = np.zeros((1,) + INPUT_SHAPE, dtype=np.float32)
    feat_a = emotion_extractor.predict(dummy, verbose=0)
    feat_b = pos_extractor.predict(dummy,     verbose=0)
    feat_dim_a = int(np.prod(feat_a.shape[1:]))
    feat_dim_b = int(np.prod(feat_b.shape[1:]))
    print(f"📊 Feature dimensions -> Emotion: {feat_dim_a}, POS: {feat_dim_b}")

    # -------------------------
    # Build model
    # -------------------------
    print("🏗️ Building balanced fusion model with neurosymbolic regularization...")
    eeg_input    = Input(shape=INPUT_SHAPE, name="eeg_input")
    lex_prior_in = Input(shape=(NUM_CLASSES_EMOTION,), name="lex_prior_input")  # zeros if unknown

    feat_a_tensor = emotion_extractor(eeg_input)
    feat_b_tensor = pos_extractor(eeg_input)

    feat_a_flat = Flatten(name="emotion_features_flat")(feat_a_tensor)
    feat_b_flat = Flatten(name="pos_features_flat")(feat_b_tensor)

    fused = create_simple_attention_fusion(feat_a_flat, feat_b_flat, fusion_dim=FUSION_DENSE)

    # Shared trunk
    x = Dense(FUSION_DENSE, activation="relu", name="fusion_dense1")(fused)
    x = BatchNormalization(name="fusion_bn1")(x)
    x = Dropout(0.3, name="fusion_dropout1")(x)
    x = Dense(FUSION_DENSE//2, activation="relu", name="fusion_dense2")(x)
    x = BatchNormalization(name="fusion_bn2")(x)
    x = Dropout(0.2, name="fusion_dropout2")(x)

    # Task heads
    emotion_out = Dense(NUM_CLASSES_EMOTION, activation="softmax", name="emotion")(x)
    pos_out     = Dense(NUM_CLASSES_POS,     activation="softmax", name="pos")(x)

    # Predicate head (latent symbols) – branch from fused features to stay EEG-grounded
    pred_base     = Dense(FUSION_DENSE//2, activation='relu', name='predicate_dense')(fused)
    predicate_out = Dense(NUM_PREDICATES, activation='sigmoid', name='predicates')(pred_base)

    # Neurosymbolic logic layer (adds loss; output is ignored)
    logic_dummy = LogicConstraintLayer()([emotion_out, pos_out, predicate_out, lex_prior_in])

    # Build final model. We only OUTPUT the supervised heads.
    combined_model = Model(
        inputs=[eeg_input, lex_prior_in],
        outputs=[emotion_out, pos_out, logic_dummy],  # Include logic_dummy
        name="balanced_fusion_model_neurosymbolic"
    )

    # -------------------------
    # DATA
    # -------------------------
    print("📂 Loading datasets...")
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data     = load_pickle(POS_DATA_PATH)

    # Emotion
    X_e = np.array(emotion_data["data"]).reshape(-1, 4, 60)
    y_e = np.array(emotion_data["labels"])
    words_e = None
    if isinstance(emotion_data, dict) and ("words" in emotion_data):
        words_e = list(emotion_data["words"])

    X_e_improved, y_e_improved, emotion_class_weights = apply_conservative_emotion_improvements(X_e, y_e)

    # POS
    X_p = np.array(pos_data["data"]).reshape(-1, 4, 60)
    X_p = robust_normalize_eeg(X_p)
    y_p = (np.array(pos_data["labels"]) - 1)

    # Optional SMOTE for POS
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

    # Add channel dim
    X_e_improved = X_e_improved[..., np.newaxis]
    X_p_balanced = X_p_balanced[..., np.newaxis]

    # Gentle channel weighting (optional)
    channel_weights = np.array([1.0, 1.0, 0.75, 0.75]).reshape(1, 4, 1, 1)
    X_e_improved = X_e_improved * channel_weights

    # Combine datasets (multi-task)
    X_combined = np.concatenate([X_e_improved, X_p_balanced], axis=0)

    y_emotion_combined = np.concatenate([
        to_categorical(y_e_improved, NUM_CLASSES_EMOTION),
        np.zeros((len(X_p_balanced), NUM_CLASSES_EMOTION), dtype=np.float32)
    ], axis=0)

    y_pos_combined = np.concatenate([
        np.zeros((len(X_e_improved), NUM_CLASSES_POS), dtype=np.float32),
        to_categorical(y_p_balanced, NUM_CLASSES_POS)
    ], axis=0)

    # Task sample weights (so each head trains on its own rows)
    sw_emotion = np.concatenate([np.ones(len(X_e_improved)), np.zeros(len(X_p_balanced))], axis=0)
    sw_pos     = np.concatenate([np.zeros(len(X_e_improved)), np.ones(len(X_p_balanced))], axis=0)

    # Lexicon priors per-sample (zeros where unknown)
    lexicon = maybe_load_valence_lexicon(VALENCE_LEXICON_PATH)
    lex_priors_emotion = None
    if words_e is not None:
        lex_priors_emotion = build_lexicon_priors_for_emotion_samples(words_e, lexicon)
    if lex_priors_emotion is None or lex_priors_emotion.shape[0] != len(X_e_improved):
        lex_priors_emotion = np.zeros((len(X_e_improved), NUM_CLASSES_EMOTION), dtype=np.float32)
    lex_priors_pos = np.zeros((len(X_p_balanced), NUM_CLASSES_EMOTION), dtype=np.float32)
    lex_priors_combined = np.concatenate([lex_priors_emotion, lex_priors_pos], axis=0).astype(np.float32)

    # Stratified split
    print("🎯 Creating stratified train/validation split...")
    train_idx, val_idx = create_stratified_split(
        X_combined, y_e_improved, y_p_balanced, test_size=0.2, random_state=RANDOM_STATE
    )

    X_train, X_val = X_combined[train_idx], X_combined[val_idx]
    ye_train, ye_val = y_emotion_combined[train_idx], y_emotion_combined[val_idx]
    yp_train, yp_val = y_pos_combined[train_idx],     y_pos_combined[val_idx]
    swe_train, swe_val = sw_emotion[train_idx], sw_emotion[val_idx]
    swp_train, swp_val = sw_pos[train_idx],           sw_pos[val_idx]
    lex_train, lex_val = lex_priors_combined[train_idx], lex_priors_combined[val_idx]

    # -------------------------
    # COMPILE & TRAIN
    # -------------------------
    print("⚙️ Compiling model (neurosymbolic regularization is added via layer.add_loss)...")
    # combined_model.compile(
    #     optimizer=Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
    #     loss=[
    #         balanced_focal_loss(alpha=0.25, gamma=1.5),  # emotion
    #         balanced_focal_loss(alpha=0.25, gamma=1.5),  # pos  
    #         lambda y_true, y_pred: 0.0  # dummy loss for logic layer (loss is added via add_loss)
    #     ],
    #     metrics=[['accuracy'], ['accuracy'], []],
    #     loss_weights=[1.2, 1.0, 0.0]  # Zero weight for dummy output
    # )
    # callbacks = [
    #     EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    #     ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
    #     ModelCheckpoint(filepath='ns_new_best_balanced_fusion_model.h5',
    #                     monitor='val_loss', save_best_only=True, verbose=1)
    # ]

    print("🏋️ Starting training...")
#     history = combined_model.fit(
#         [X_train, lex_train],
#         [ye_train, yp_train, np.zeros((len(X_train), NUM_CLASSES_EMOTION))],  # Add dummy
#         sample_weight=[swe_train, swp_train, np.zeros(len(X_train))],  # Add zero weights
#         validation_data=([X_val, lex_val], 
#                         [ye_val, yp_val, np.zeros((len(X_val), NUM_CLASSES_EMOTION))], 
#                         [swe_val, swp_val, np.zeros(len(X_val))]),
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         callbacks=callbacks,
#         verbose=1
# )

    # -------------------------
    # EVALUATION
    # -------------------------
    print("📊 Evaluation...")
    best_model = load_model(
        'ns_new_best_balanced_fusion_model.h5',
        custom_objects={'focal_loss_fixed': balanced_focal_loss(), 'LogicConstraintLayer': LogicConstraintLayer},
        compile=False
    ) 

    y_pred_emotion, y_pred_pos, logic_dummy_pred = best_model.predict([X_val, lex_val], verbose=1)
    y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
    y_pred_pos_classes     = np.argmax(y_pred_pos, axis=1)

    ye_val_labels = np.argmax(ye_val, axis=1)
    yp_val_labels = np.argmax(yp_val, axis=1)

    emotion_mask = swe_val > 0
    if np.sum(emotion_mask) > 0:
        print("\n🎭 EMOTION CLASSIFICATION RESULTS:")
        print(classification_report(
            ye_val_labels[emotion_mask],
            y_pred_emotion_classes[emotion_mask],
            target_names=["Negative", "Neutral", "Positive"]
        ))
        emotion_cm = confusion_matrix(ye_val_labels[emotion_mask], y_pred_emotion_classes[emotion_mask])
        print("Emotion Confusion Matrix:")
        print(emotion_cm)
        per_class = np.diag(emotion_cm) / np.sum(emotion_cm, axis=1) * 100
        print(f"Per-class accuracy: Negative: {per_class[0]:.1f}%, Neutral: {per_class[1]:.1f}%, Positive: {per_class[2]:.1f}%")

    pos_mask = swp_val > 0
    if np.sum(pos_mask) > 0:
        print("\n📝 POS CLASSIFICATION RESULTS:")
        print(classification_report(
            yp_val_labels[pos_mask],
            y_pred_pos_classes[pos_mask],
            target_names=["Noun", "Verb"]
        ))
    # Generate timestamp for file naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot confusion matrices
    print("\n📈 Generating confusion matrix plots...")
    confusion_fig = plot_confusion_matrices(
        ye_val_labels, y_pred_emotion_classes,
        yp_val_labels, y_pred_pos_classes,
        emotion_mask, pos_mask,
        save_path=f'confusion_matrices_neurosymbolic_best_model_2_further_improved{timestamp}.png'
    )

    print("\n✅ Training complete!")
    print("📁 Best model: third_weighted_best_balanced_fusion_model.h5")
    return best_model

if __name__ == "__main__":
    model, training_history = main()
