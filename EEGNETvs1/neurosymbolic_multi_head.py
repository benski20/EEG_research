# improved_multi_task_fusion_balanced.py - BALANCED APPROACH + NEUROSYMBOLIC UPGRADES
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Concatenate, Dropout, BatchNormalization, Flatten,
    Lambda
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
# BALANCED CONFIG - CONSERVATIVE IMPROVEMENTS
# -------------------------
EMOTION_MODEL_PATH = "eegnet_fft_model_1_regular_eegarch_12500samples_20250802-0132.h5"
POS_MODEL_PATH     = "eegnet_noun_verb_averagepooling_50epochs_regular_arch_12500samples20250805-2223.h5"
EMOTION_DATA_PATH  = "eeg_fft_dataset.pkl"
POS_DATA_PATH      = "eeg_fft_nounvsverb_JASON_dataset.pkl"
VALENCE_LEXICON_PATH = "valence_lexicon.pkl"  # optional; dict: word -> [p_neg, p_neu, p_pos]

INPUT_SHAPE = (4, 60, 1)
NUM_CLASSES_EMOTION = 3  # [Negative, Neutral, Positive]
NUM_CLASSES_POS     = 2  # [Noun, Verb]
FUSION_DENSE = 256
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
RANDOM_STATE = 42

# -------------------------
# NEUROSYMBOLIC HYPERPARAMETERS
# -------------------------
NUM_PREDICATES = 5  # [FA_Left, FA_Right, OccipitalAlphaDrop, MotorBetaLeft, MotorBetaRight]

LMB_SEMANTIC = 1     # lexicon prior KL (gated by confidence)
LMB_LOGIC    = 1    # fuzzy rule penalties
LMB_COMPAT   = 1     # PoS↔Valence compatibility prior
LMB_PRED_REG = 0.1    # regularize predicate activations

CONF_TAU = 0.65        # confidence threshold for lexicon gating (use prior if model is uncertain)
PRED_TARGET_MEAN = 0.20  # gentle target mean activation for predicates (avoid trivial 0s)

APPLY_POSTHOC = True   # apply snapper after prediction for an extra consistency pass

# -------------------------
# GENTLE FOCAL LOSS - BALANCED APPROACH
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
# DATA HELPERS (unchanged + small utilities)
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
        if len(model.layers) >= 2:
            feature_output = model.layers[-2].output
        else:
            raise ValueError("Could not identify a feature layer in model.")
    feature_extractor = Model(inputs=model.input, outputs=feature_output)
    return feature_extractor

def robust_normalize_eeg(X):
    for ch in range(X.shape[1]):
        channel_data = X[:, ch, :]
        p5  = np.percentile(channel_data, 5,  axis=1, keepdims=True)
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

# -------------------------
# CONSERVATIVE EMOTION IMPROVEMENTS (unchanged)
# -------------------------
def gentle_emotion_preprocessing(X_emotion):
    X_processed = X_emotion.copy().astype(np.float32)
    for ch in range(X_processed.shape[1]):
        channel_data = X_processed[:, ch, :]
        p25 = np.percentile(channel_data, 25, axis=1, keepdims=True)
        p75 = np.percentile(channel_data, 75, axis=1, keepdims=True)
        scale = p75 - p25
        scale = np.where(scale == 0, 1, scale)
        X_processed[:, ch, :] = (channel_data - p25) / scale
    return X_processed

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
        aug_sample = sample.copy()
        noise_std = 0.01 * np.std(aug_sample)
        noise = np.random.normal(0, noise_std, aug_sample.shape)
        aug_sample += noise
        neutral_augmented.append(aug_sample)

    if len(neutral_augmented) > 0:
        augmented_samples.append(np.array(neutral_augmented))
        augmented_labels.append(np.full(len(neutral_augmented), neutral_class))

    X_final = np.concatenate(augmented_samples, axis=0)
    y_final = np.concatenate(augmented_labels, axis=0)
    print(f"🎭 After conservative augmentation: {np.bincount(y_final)}")
    return X_final, y_final

def conservative_class_weights(y_emotion):
    classes = np.unique(y_emotion)
    standard_weights = compute_class_weight('balanced', classes=classes, y=y_emotion)
    emotion_weights = dict(zip(classes, standard_weights))
    if 1 in emotion_weights:
        emotion_weights[1] *= 1.3
    print(f"🎯 Conservative class weights: {emotion_weights}")
    return emotion_weights

def apply_conservative_emotion_improvements(X_emotion, y_emotion):
    print("🎯 Applying conservative emotion improvements...")
    print("🔧 Light preprocessing...")
    X_processed = gentle_emotion_preprocessing(X_emotion)

    print("🎭 Conservative neutral augmentation...")
    X_augmented, y_augmented = conservative_neutral_augmentation(X_processed, y_emotion)

    print("⚖️ Conservative SMOTE...")
    original_shape = X_augmented.shape
    X_flat = X_augmented.reshape(X_augmented.shape[0], -1)

    class_counts = np.bincount(y_augmented)
    min_count = np.min(class_counts)
    max_count = np.max(class_counts)

    if max_count > min_count * 1.5:
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_flat, y_augmented = smote.fit_resample(X_flat, y_augmented)
        X_balanced = X_flat.reshape(-1, *original_shape[1:])
    else:
        print("📊 Classes reasonably balanced, skipping SMOTE")
        X_balanced = X_augmented

    class_weights = conservative_class_weights(y_augmented)

    print(f"📊 Final emotion dataset shape: {X_balanced.shape}")
    print(f"📊 Final emotion distribution: {np.bincount(y_augmented)}")
    return X_balanced, y_augmented, class_weights

# -------------------------
# SIMPLE FUSION ARCHITECTURE (unchanged base)
# -------------------------
def create_simple_attention_fusion(emotion_features, pos_features, fusion_dim=256):
    emotion_proj = Dense(fusion_dim//2, activation='relu', name='emotion_proj')(emotion_features)
    pos_proj     = Dense(fusion_dim//2, activation='relu', name='pos_proj')(pos_features)

    emotion_attention = Dense(1, activation='sigmoid', name='emotion_att')(emotion_proj)
    pos_attention     = Dense(1, activation='sigmoid', name='pos_att')(pos_proj)

    emotion_weighted = Lambda(lambda x: x[0] * x[1])([emotion_proj, emotion_attention])
    pos_weighted     = Lambda(lambda x: x[0] * x[1])([pos_proj, pos_attention])

    fused = Concatenate()([emotion_weighted, pos_weighted])
    return fused

# ============================================================
# 🔹 NEW: Logic layer (differentiable rules + priors + compat)
# ============================================================
import tensorflow as tf

class LogicConstraintLayer(tf.keras.layers.Layer):
    def __init__(self,
                 num_val=NUM_CLASSES_EMOTION,
                 num_pos=NUM_CLASSES_POS,
                 lmb_sem=LMB_SEMANTIC,
                 lmb_logic=LMB_LOGIC,
                 lmb_compat=LMB_COMPAT,
                 lmb_pred_reg=LMB_PRED_REG,
                 conf_tau=CONF_TAU,
                 pred_target_mean=PRED_TARGET_MEAN,
                 name="logic_constraints",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_val = num_val
        self.num_pos = num_pos
        self.lmb_sem = lmb_sem
        self.lmb_logic = lmb_logic
        self.lmb_compat = lmb_compat
        self.lmb_pred_reg = lmb_pred_reg
        self.conf_tau = conf_tau
        self.pred_target_mean = pred_target_mean

        # learned PoS↔Val compatibility logits
        self.compat_logits = self.add_weight(
            name="compat_logits",
            shape=(self.num_pos * self.num_val,),
            initializer="zeros",
            trainable=True
        )

    @staticmethod
    def _kl(p, q, eps=1e-6):
        p = tf.clip_by_value(p, eps, 1.0)
        q = tf.clip_by_value(q, eps, 1.0)
        return tf.reduce_sum(p * (tf.math.log(p) - tf.math.log(q)), axis=-1)

    def call(self, inputs):
        e_probs, p_probs, pred_probs, lex_prior, mask_e = inputs
        eps = 1e-6

        # ------------------ Semantic loss ------------------
        has_prior = tf.cast(tf.reduce_sum(lex_prior, axis=-1, keepdims=True) > 0.0, tf.float32)
        kl_sem = self._kl(e_probs, lex_prior)
        loss_semantic = tf.reduce_mean(kl_sem * tf.squeeze(has_prior, -1)) * self.lmb_sem

        # ------------------ Fuzzy logic rules ------------------
        fa_left  = pred_probs[:, 0]
        fa_right = pred_probs[:, 1]
        val_neg = e_probs[:, 0]
        val_pos = e_probs[:, 2]

        truth_r1 = tf.minimum(1.0, 1.0 - fa_left  + val_pos)
        truth_r2 = tf.minimum(1.0, 1.0 - fa_right + val_neg)
        viol_r1  = 1.0 - truth_r1
        viol_r2  = 1.0 - truth_r2
        loss_logic = tf.reduce_mean(0.5 * (viol_r1 + viol_r2)) * self.lmb_logic

        # ------------------ PoS↔Valence compatibility ------------------
        p_joint = tf.einsum('bi,bj->bij', p_probs, e_probs)  # (B,2,3)
        compat_probs = tf.nn.softmax(self.compat_logits)
        compat_probs = tf.reshape(compat_probs, (self.num_pos, self.num_val))
        kl_joint = tf.reduce_sum(p_joint * (tf.math.log(tf.clip_by_value(p_joint, eps, 1.0))
                                            - tf.math.log(tf.clip_by_value(compat_probs, eps, 1.0))),
                                 axis=[1,2])
        loss_compat = tf.reduce_mean(kl_joint) * self.lmb_compat

        # ------------------ Predicate regularization ------------------
        mean_pred = tf.reduce_mean(pred_probs, axis=-1)
        loss_pred_reg = tf.reduce_mean(tf.square(mean_pred - self.pred_target_mean)) * self.lmb_pred_reg

        total_loss = loss_semantic + loss_logic + loss_compat + loss_pred_reg

        # Add as layer loss (used in model.compile)
        self.add_loss(total_loss)

        # Return e_probs (known shape) so Keras can build the graph
        return e_probs


# -------------------------
# Optional: load lexicon
# -------------------------
def maybe_load_valence_lexicon(path):
    if os.path.exists(path):
        try:
            d = load_pickle(path)
            if isinstance(d, dict):
                return d
        except Exception:
            pass
    return None  # quietly skip if unavailable

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

# -------------------------
# POST-HOC REASONING (snapper)
# -------------------------
def posthoc_snap(emotion_probs, pos_probs, predicate_probs, lex_priors=None,
                 w_r1=0.6, w_r2=0.6, conf_tau=CONF_TAU):
    """
    One light-weight factor update:
    - Multiply emotion probs by exp(weight * predicate) for rules, renormalize
    - Optionally blend lexicon prior when model confidence is low
    """
    emo = emotion_probs.copy()
    pos = pos_probs.copy()
    preds = predicate_probs.copy()

    fa_left  = preds[:, 0]  # -> Positive
    fa_right = preds[:, 1]  # -> Negative

    # Rule boosts
    emo[:, 2] *= np.exp(w_r1 * fa_left)   # Positive
    emo[:, 0] *= np.exp(w_r2 * fa_right)  # Negative

    # Renormalize
    emo = emo / (emo.sum(axis=1, keepdims=True) + 1e-8)

    # Confidence-gated lexicon blend
    if lex_priors is not None:
        conf = emo.max(axis=1, keepdims=True)
        has_prior = (lex_priors.sum(axis=1, keepdims=True) > 0).astype(np.float32)
        alpha = ((conf < conf_tau) * has_prior).astype(np.float32)  # 1 if blend
        # simple convex blend: new = normalize(emo^(1-alpha) * prior^alpha)
        blended = (emo ** (1 - alpha)) * (np.clip(lex_priors, 1e-6, 1.0) ** alpha)
        blended = blended / (blended.sum(axis=1, keepdims=True) + 1e-8)
        emo = blended

    return emo, pos

# -------------------------
# MAIN EXECUTION
# -------------------------
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
    # Build fusion + heads + predicates
    # -------------------------
    print("🏗️ Building balanced fusion model with neurosymbolic components...")
    eeg_input      = Input(shape=INPUT_SHAPE, name="eeg_input")
    lex_prior_in   = Input(shape=(NUM_CLASSES_EMOTION,), name="lex_prior_input")     # zeros if unknown
    mask_emotion_in= Input(shape=(1,), name="mask_emotion_input")                    # 1 for emotion samples else 0

    feat_a_tensor = emotion_extractor(eeg_input)
    feat_b_tensor = pos_extractor(eeg_input)

    feat_a_flat = Flatten(name="emotion_features_flat")(feat_a_tensor)
    feat_b_flat = Flatten(name="pos_features_flat")(feat_b_tensor)

    fused_features = create_simple_attention_fusion(feat_a_flat, feat_b_flat, fusion_dim=FUSION_DENSE)

    # shared fusion trunk
    x = Dense(FUSION_DENSE, activation="relu", name="fusion_dense1")(fused_features)
    x = BatchNormalization(name="fusion_bn1")(x)
    x = Dropout(0.3, name="fusion_dropout1")(x)
    x = Dense(FUSION_DENSE//2, activation="relu", name="fusion_dense2")(x)
    x = BatchNormalization(name="fusion_bn2")(x)
    x = Dropout(0.2, name="fusion_dropout2")(x)

    # heads
    emotion_out = Dense(NUM_CLASSES_EMOTION, activation="softmax", name="emotion")(x)
    pos_out     = Dense(NUM_CLASSES_POS,     activation="softmax", name="pos")(x)

    # 🔹 predicate head (branch off earlier features to stay EEG-grounded)
    pred_base   = Dense(FUSION_DENSE//2, activation='relu', name='predicate_dense')(fused_features)
    predicate_out = Dense(NUM_PREDICATES, activation='sigmoid', name='predicates')(pred_base)

    # 🔹 logic constraints layer (adds loss; output is dummy to keep layer alive)
    logic_dummy = LogicConstraintLayer()([emotion_out, pos_out, predicate_out, lex_prior_in, mask_emotion_in])

    # Include dummy + predicate output as model outputs (no losses) so we can fetch predicates later
    combined_model = Model(
        inputs=[eeg_input, lex_prior_in, mask_emotion_in],
        outputs=[emotion_out, pos_out, logic_dummy, predicate_out],
        name="balanced_fusion_model_neurosymbolic"
    )

    # -------------------------
    # DATA LOADING AND PREPROCESSING
    # -------------------------
    print("📂 Loading datasets...")
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data     = load_pickle(POS_DATA_PATH)

    # Emotion data
    X_e = np.array(emotion_data["data"]).reshape(-1, 4, 60)
    y_e = np.array(emotion_data["labels"])
    words_e = None
    if isinstance(emotion_data, dict) and ("words" in emotion_data):
        words_e = list(emotion_data["words"])

    # Apply conservative improvements
    X_e_improved, y_e_improved, emotion_class_weights = apply_conservative_emotion_improvements(X_e, y_e)

    # POS data
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

    # (Keep your channel reweighting if desired)
    channel_weights = np.array([1.0, 1.0, 0.75, 0.75]).reshape(1, 4, 1)
    X_e = X_e * channel_weights  # original (not used later); retained for compatibility with your code

    # Combine datasets
    X_combined = np.concatenate([X_e_improved, X_p_balanced], axis=0)

    y_emotion_combined = np.concatenate([
        to_categorical(y_e_improved, NUM_CLASSES_EMOTION),
        np.zeros((len(X_p_balanced), NUM_CLASSES_EMOTION), dtype=np.float32)
    ], axis=0)

    y_pos_combined = np.concatenate([
        np.zeros((len(X_e_improved), NUM_CLASSES_POS), dtype=np.float32),
        to_categorical(y_p_balanced, NUM_CLASSES_POS)
    ], axis=0)

    # Sample/task masks (reuse your sample weights)
    mask_emotion = np.concatenate([np.ones((len(X_e_improved), 1), dtype=np.float32),
                                   np.zeros((len(X_p_balanced), 1), dtype=np.float32)], axis=0)

    # Lexicon priors per-sample (zeros if not available)
    lexicon = maybe_load_valence_lexicon(VALENCE_LEXICON_PATH)
    lex_priors_emotion = None
    if words_e is not None:
        lex_priors_emotion = build_lexicon_priors_for_emotion_samples(words_e, lexicon)

    if lex_priors_emotion is None or lex_priors_emotion.shape[0] != len(X_e_improved):
        # fill with zeros to indicate 'no prior' (layer will skip)
        lex_priors_emotion = np.zeros((len(X_e_improved), NUM_CLASSES_EMOTION), dtype=np.float32)

    lex_priors_pos = np.zeros((len(X_p_balanced), NUM_CLASSES_EMOTION), dtype=np.float32)
    lex_priors_combined = np.concatenate([lex_priors_emotion, lex_priors_pos], axis=0).astype(np.float32)

    # Sample weights for task-specific heads
    sw_emotion = np.concatenate([np.ones(len(X_e_improved)), np.zeros(len(X_p_balanced))], axis=0)
    sw_pos     = np.concatenate([np.zeros(len(X_e_improved)), np.ones(len(X_p_balanced))], axis=0)

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
    mask_train, mask_val = mask_emotion[train_idx], mask_emotion[val_idx]

    # -------------------------
    # Compile & Train
    # -------------------------
    print("⚙️ Compiling model with focal + neurosymbolic add-losses...")
    combined_model.compile(
        optimizer=Adam(learning_rate=LR, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
        loss=[
            balanced_focal_loss(alpha=0.25, gamma=1.5),  # emotion
            balanced_focal_loss(alpha=0.25, gamma=1.5),  # pos
            None,                                        # logic_dummy (no direct loss)
            None                                         # predicates (no direct loss)
        ],
        metrics=[['accuracy'], ['accuracy'], None, None],
        loss_weights=[1.2, 1.0, 0.0, 0.0]
    )

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, min_lr=1e-7, verbose=1),
        ModelCheckpoint(filepath='neurosymbolic_fusion_model_more_weight_influence_with_lexicon.h5',
                        monitor='val_loss', save_best_only=True, verbose=1)
    ]

    print("🏋️ Starting balanced training...")
    history = combined_model.fit(
        [X_train, lex_train, mask_train],
        [ye_train, yp_train],
        sample_weight=[swe_train, swp_train],
        validation_data=([X_val, lex_val, mask_val], [ye_val, yp_val], [swe_val, swp_val]),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    # -------------------------
    # EVALUATION
    # -------------------------
    print("📊 Evaluation...")

    best_model = load_model(
        'neurosymbolic_fusion_model_more_weight_influence_with_lexicon.h5',
        custom_objects={
            'focal_loss_fixed': balanced_focal_loss(),
            'LogicConstraintLayer': LogicConstraintLayer
        }
    )

    # Fetch outputs: [emotion, pos, logic_dummy, predicates]
    y_pred_emotion, y_pred_pos, _, y_pred_predicates = best_model.predict([X_val, lex_val, mask_val], verbose=0)
    y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
    y_pred_pos_classes     = np.argmax(y_pred_pos, axis=1)

    ye_val_labels = np.argmax(ye_val, axis=1)
    yp_val_labels = np.argmax(yp_val, axis=1)

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
        emotion_per_class_acc = np.diag(emotion_cm) / np.sum(emotion_cm, axis=1) * 100
        print(f"Per-class accuracy: Negative: {emotion_per_class_acc[0]:.1f}%, Neutral: {emotion_per_class_acc[1]:.1f}%, Positive: {emotion_per_class_acc[2]:.1f}%")

    pos_mask = swp_val > 0
    if np.sum(pos_mask) > 0:
        print("\n📝 POS CLASSIFICATION RESULTS:")
        print(classification_report(
            yp_val_labels[pos_mask],
            y_pred_pos_classes[pos_mask],
            target_names=["Noun", "Verb"]
        ))

    # -------------------------
    # Optional: post-hoc reasoning snapper
    # -------------------------
    if APPLY_POSTHOC:
        print("\n🔧 Applying post-hoc neurosymbolic snapper...")
        lex_val_emotion_only = lex_val  # already aligned; zeros where unknown
        emo_snap, pos_snap = posthoc_snap(y_pred_emotion, y_pred_pos, y_pred_predicates, lex_val_emotion_only)

        y_snap_classes = np.argmax(emo_snap, axis=1)

        if np.sum(emotion_mask) > 0:
            print("\n🎭 EMOTION (SNAPPED) RESULTS:")
            print(classification_report(
                ye_val_labels[emotion_mask],
                y_snap_classes[emotion_mask],
                target_names=["Negative", "Neutral", "Positive"]
            ))

        # Simple human-readable explanations for first 5 emotion samples
        idxs = np.where(emotion_mask)[0][:5]
        for i in idxs:
            fa_left, fa_right = y_pred_predicates[i, 0], y_pred_predicates[i, 1]
            conf_before = y_pred_emotion[i].max()
            conf_after  = emo_snap[i].max()
            chosen = ["Negative", "Neutral", "Positive"][np.argmax(emo_snap[i])]
            fired = []
            if fa_left  > 0.5:  fired.append("FA_Left ⇒ Positive")
            if fa_right > 0.5:  fired.append("FA_Right ⇒ Negative")
            fired_str = "; ".join(fired) if fired else "no strong rules"
            print(f"💬 Sample {i}: snapped to {chosen} (conf {conf_before:.2f}→{conf_after:.2f}); rules: {fired_str}")

    print("\n✅ Balanced multi-task fusion training completed!")
    print(f"📁 Best model saved as: third_weighted_best_balanced_fusion_model.h5")
    return best_model, history

if __name__ == "__main__":
    model, training_history = main()
