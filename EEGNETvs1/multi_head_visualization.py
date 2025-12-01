# visualization_only.py - Just load model and create confusion matrices
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# CONFIG
# -------------------------
EMOTION_DATA_PATH = "eeg_fft_dataset.pkl"
POS_DATA_PATH = "eeg_fft_nounvsverb_JASON_dataset.pkl"
MODEL_PATH = "third_weighted_best_balanced_fusion_model.h5"

NUM_CLASSES_EMOTION = 3
NUM_CLASSES_POS = 2
RANDOM_STATE = 42

# -------------------------
# FOCAL LOSS FOR MODEL LOADING
# -------------------------
def balanced_focal_loss(alpha=0.25, gamma=1.5):
    """Focal loss function needed for loading the model"""
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
# DATA PREPROCESSING FUNCTIONS (simplified versions)
# -------------------------
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def gentle_emotion_preprocessing(X_emotion):
    """Light preprocessing"""
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
    """Conservative augmentation"""
    neutral_mask = (y_emotion == neutral_class)
    neutral_samples = X_emotion[neutral_mask]
    
    if len(neutral_samples) == 0:
        return X_emotion, y_emotion
    
    augmented_samples = [X_emotion]
    augmented_labels = [y_emotion]
    
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
    
    return X_final, y_final

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

def apply_conservative_emotion_improvements(X_emotion, y_emotion):
    """Apply the same preprocessing used during training"""
    print("🎯 Applying conservative emotion improvements...")
    
    X_processed = gentle_emotion_preprocessing(X_emotion)
    X_augmented, y_augmented = conservative_neutral_augmentation(X_processed, y_emotion)
    
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
        X_balanced = X_augmented
    
    return X_balanced, y_augmented

# -------------------------
# VISUALIZATION FUNCTIONS
# -------------------------
def plot_confusion_matrix_with_accuracy(y_true, y_pred, class_names, title_prefix="", 
                                       figsize=(8, 6), cmap='Blues', save_path=None):
    """Plot confusion matrix with overall and per-class accuracies in the title"""
    cm = confusion_matrix(y_true, y_pred)
    overall_acc = accuracy_score(y_true, y_pred) * 100
    per_class_acc = np.diag(cm) / np.sum(cm, axis=1) * 100
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    per_class_str = ", ".join([f"{name}: {acc:.1f}%" for name, acc in zip(class_names, per_class_acc)])
    title = f"{title_prefix}Confusion Matrix\nOverall Accuracy: {overall_acc:.1f}% | Per-class: {per_class_str}"
    
    plt.title(title, fontsize=12, pad=20)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.ylabel('True Label', fontsize=11)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return cm, overall_acc, per_class_acc

def visualize_model_performance(model_path, X_val, ye_val, yp_val, swe_val, swp_val):
    """Load model and visualize confusion matrices for both tasks"""
    print("🔄 Loading model...")
    model = load_model(model_path, custom_objects={'focal_loss_fixed': balanced_focal_loss()})
    
    print("🔮 Making predictions...")
    y_pred_emotion, y_pred_pos = model.predict(X_val, verbose=0)
    
    y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
    y_pred_pos_classes = np.argmax(y_pred_pos, axis=1)
    ye_val_labels = np.argmax(ye_val, axis=1)
    yp_val_labels = np.argmax(yp_val, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    cm_emotion = None
    cm_pos = None
    overall_acc_emotion = None
    overall_acc_pos = None
    
    # Emotion task visualization
    emotion_mask = swe_val > 0
    if np.sum(emotion_mask) > 0:
        print("🎭 Visualizing emotion classification results...")
        
        y_true_emotion = ye_val_labels[emotion_mask]
        y_pred_emotion_filtered = y_pred_emotion_classes[emotion_mask]
        
        cm_emotion = confusion_matrix(y_true_emotion, y_pred_emotion_filtered)
        overall_acc_emotion = accuracy_score(y_true_emotion, y_pred_emotion_filtered) * 100
        per_class_acc_emotion = np.diag(cm_emotion) / np.sum(cm_emotion, axis=1) * 100
        
        plt.sca(axes[0])
        emotion_classes = ["Negative", "Neutral", "Positive"]
        sns.heatmap(cm_emotion, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=emotion_classes, yticklabels=emotion_classes,
                    cbar_kws={'label': 'Count'})
        
        per_class_str_emotion = ", ".join([f"{name}: {acc:.1f}%" for name, acc in zip(emotion_classes, per_class_acc_emotion)])
        title_emotion = f"Emotion Classification\nOverall Accuracy: {overall_acc_emotion:.1f}%\nPer-class: {per_class_str_emotion}"
        
        plt.title(title_emotion, fontsize=11, pad=15)
        plt.xlabel('Predicted Label', fontsize=10)
        plt.ylabel('True Label', fontsize=10)
        
        print(f"📊 Emotion Task - Overall: {overall_acc_emotion:.1f}%, Per-class: {per_class_str_emotion}")
    
    # POS task visualization
    pos_mask = swp_val > 0
    if np.sum(pos_mask) > 0:
        print("📝 Visualizing POS classification results...")
        
        y_true_pos = yp_val_labels[pos_mask]
        y_pred_pos_filtered = y_pred_pos_classes[pos_mask]
        
        cm_pos = confusion_matrix(y_true_pos, y_pred_pos_filtered)
        overall_acc_pos = accuracy_score(y_true_pos, y_pred_pos_filtered) * 100
        per_class_acc_pos = np.diag(cm_pos) / np.sum(cm_pos, axis=1) * 100
        
        plt.sca(axes[1])
        pos_classes = ["Noun", "Verb"]
        sns.heatmap(cm_pos, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=pos_classes, yticklabels=pos_classes,
                    cbar_kws={'label': 'Count'})
        
        per_class_str_pos = ", ".join([f"{name}: {acc:.1f}%" for name, acc in zip(pos_classes, per_class_acc_pos)])
        title_pos = f"POS Classification\nOverall Accuracy: {overall_acc_pos:.1f}%\nPer-class: {per_class_str_pos}"
        
        plt.title(title_pos, fontsize=11, pad=15)
        plt.xlabel('Predicted Label', fontsize=10)
        plt.ylabel('True Label', fontsize=10)
        
        print(f"📊 POS Task - Overall: {overall_acc_pos:.1f}%, Per-class: {per_class_str_pos}")
    
    plt.tight_layout()
    plt.savefig('plots/third_weighted_confusion_matrices_balanced_fusion.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return {
        'emotion_cm': cm_emotion,
        'pos_cm': cm_pos,
        'emotion_acc': overall_acc_emotion,
        'pos_acc': overall_acc_pos
    }

def plot_individual_confusion_matrices(model_path, X_val, ye_val, yp_val, swe_val, swp_val):
    """Create separate, larger confusion matrix plots for each task"""
    model = load_model(model_path, custom_objects={'focal_loss_fixed': balanced_focal_loss()})
    y_pred_emotion, y_pred_pos = model.predict(X_val, verbose=0)
    
    y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
    y_pred_pos_classes = np.argmax(y_pred_pos, axis=1)
    ye_val_labels = np.argmax(ye_val, axis=1)
    yp_val_labels = np.argmax(yp_val, axis=1)
    
    # Individual emotion plot
    emotion_mask = swe_val > 0
    if np.sum(emotion_mask) > 0:
        y_true_emotion = ye_val_labels[emotion_mask]
        y_pred_emotion_filtered = y_pred_emotion_classes[emotion_mask]
        emotion_classes = ["Negative", "Neutral", "Positive"]
        
        plot_confusion_matrix_with_accuracy(
            y_true_emotion, y_pred_emotion_filtered, emotion_classes,
            title_prefix="Emotion Task - ", figsize=(10, 8), cmap='Reds',
            save_path='plots/third_weighted_emotion_combined_confusion_matrix.png'
        )
    
    # Individual POS plot
    pos_mask = swp_val > 0
    if np.sum(pos_mask) > 0:
        y_true_pos = yp_val_labels[pos_mask]
        y_pred_pos_filtered = y_pred_pos_classes[pos_mask]
        pos_classes = ["Noun", "Verb"]
        
        plot_confusion_matrix_with_accuracy(
            y_true_pos, y_pred_pos_filtered, pos_classes,
            title_prefix="POS Task - ", figsize=(8, 6), cmap='Blues',
            save_path='plots/third_pos_combined_with_emotion_weighted_confusion_matrix.png'
        )

# -------------------------
# MAIN FUNCTION - LOAD DATA AND VISUALIZE ONLY
# -------------------------
def main():
    print("📂 Loading datasets...")
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data = load_pickle(POS_DATA_PATH)
    
    # Process emotion data (same preprocessing as during training)
    X_e = np.array(emotion_data["data"]).reshape(-1, 4, 60)
    y_e = np.array(emotion_data["labels"])
    X_e_improved, y_e_improved = apply_conservative_emotion_improvements(X_e, y_e)
    
    # Process POS data
    X_p = np.array(pos_data["data"]).reshape(-1, 4, 60)
    X_p = robust_normalize_eeg(X_p)
    y_p = np.array(pos_data["labels"]) - 1
    
    # Apply SMOTE to POS if needed (same as training)
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
    
    # Combine datasets
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
    
    # Create same train/validation split as used during training
    print("🎯 Creating stratified train/validation split...")
    train_idx, val_idx = create_stratified_split(
        X_combined, y_e_improved, y_p_balanced, test_size=0.2, random_state=RANDOM_STATE
    )
    
    X_val = X_combined[val_idx]
    ye_val = y_emotion_combined[val_idx]
    yp_val = y_pos_combined[val_idx]
    swe_val = sw_emotion[val_idx]
    swp_val = sw_pos[val_idx]
    
    print(f"📊 Validation set size: {len(X_val)}")
    print(f"📊 Emotion samples in validation: {np.sum(swe_val > 0)}")
    print(f"📊 POS samples in validation: {np.sum(swp_val > 0)}")
    
    # -------------------------
    # VISUALIZATION ONLY
    # -------------------------
    print("🎨 Creating confusion matrix visualizations...")
    
    # Create combined visualization
    results = visualize_model_performance(
        model_path=MODEL_PATH,
        X_val=X_val, 
        ye_val=ye_val, 
        yp_val=yp_val, 
        swe_val=swe_val, 
        swp_val=swp_val
    )
    
    # Create individual larger plots
    plot_individual_confusion_matrices(
        model_path=MODEL_PATH,
        X_val=X_val, 
        ye_val=ye_val, 
        yp_val=yp_val, 
        swe_val=swe_val, 
        swp_val=swp_val
    )
    
    print("\n✅ Confusion matrix visualization completed!")
    print(f"📊 Saved visualizations:")
    print(f"   - confusion_matrices_balanced_fusion.png (combined)")
    print(f"   - emotion_confusion_matrix.png (individual)")
    print(f"   - pos_confusion_matrix.png (individual)")
    
    return results

if __name__ == "__main__":
    viz_results = main()
        
print("✅ Visualization functions ready!")
print("💡 To use: call visualize_model_performance() or plot_individual_confusion_matrices() with your data")