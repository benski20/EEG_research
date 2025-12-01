#!/usr/bin/env python3
"""
Enhanced Multi-Task Fusion Model Evaluation Script
Loads trained model and provides comprehensive evaluation with visualizations
"""

import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# CONFIGURATION
# -------------------------
MODEL_PATH = "best_enhanced_fusion_model.h5"  # Path to your trained model
EMOTION_DATA_PATH = "eeg_fft_dataset.pkl"
POS_DATA_PATH = "eeg_fft_nounvsverb_JASON_dataset.pkl"

INPUT_SHAPE = (4, 60, 1)
NUM_CLASSES_EMOTION = 3
NUM_CLASSES_POS = 2
RANDOM_STATE = 42

# Class labels for better visualization
EMOTION_LABELS = ['Negative', 'Neutral', 'Positive']  # Adjust based on your actual labels
POS_LABELS = ['Noun', 'Verb']  # Adjust based on your actual labels

# -------------------------
# FOCAL LOSS (needed for loading model)
# -------------------------
def focal_loss(alpha=0.25, gamma=2.0):
    """Focal Loss implementation - needed for model loading"""
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
# DATA PREPROCESSING
# -------------------------
def robust_normalize_eeg(X):
    """Robust normalization using percentile-based scaling"""
    for ch in range(X.shape[1]):
        channel_data = X[:, ch, :]
        p5 = np.percentile(channel_data, 5, axis=1, keepdims=True)
        p95 = np.percentile(channel_data, 95, axis=1, keepdims=True)
        
        scale = p95 - p5
        scale = np.where(scale == 0, 1, scale)
        X[:, ch, :] = (channel_data - p5) / scale
    
    return X

def load_pickle(path):
    """Load pickle file"""
    with open(path, "rb") as f:
        return pickle.load(f)

# -------------------------
# VISUALIZATION FUNCTIONS
# -------------------------
def plot_confusion_matrix_with_accuracy(cm, labels, title, ax):
    """Create confusion matrix with accuracy percentages"""
    # Calculate accuracies
    total_samples = np.sum(cm)
    accuracy_per_class = np.diag(cm) / np.sum(cm, axis=1) * 100
    overall_accuracy = np.trace(cm) / total_samples * 100
    
    # Create heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels, ax=ax)
    
    # Add custom annotations with counts and percentages
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            if i == j:  # Diagonal (correct predictions)
                percentage = accuracy_per_class[i]
                text = f'{count}\n({percentage:.1f}%)'
                color = 'white' if count > cm.max() * 0.5 else 'black'
            else:  # Off-diagonal (incorrect predictions)
                percentage = (count / np.sum(cm[i, :])) * 100 if np.sum(cm[i, :]) > 0 else 0
                text = f'{count}\n({percentage:.1f}%)'
                color = 'white' if count > cm.max() * 0.5 else 'black'
            
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center', 
                   fontsize=10, fontweight='bold', color=color)
    
    ax.set_title(f'{title}\nOverall Accuracy: {overall_accuracy:.2f}%', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)

def create_evaluation_report(y_true, y_pred, labels, task_name):
    """Create detailed evaluation report"""
    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{task_name.upper()} TASK EVALUATION")
    print(f"{'='*50}")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    
    # Per-class accuracy
    print(f"\nPer-Class Accuracy:")
    class_accuracies = np.diag(cm) / np.sum(cm, axis=1)
    for i, label in enumerate(labels):
        print(f"  {label}: {class_accuracies[i]:.4f} ({class_accuracies[i]*100:.2f}%)")
    
    return accuracy, cm

# -------------------------
# MAIN EVALUATION FUNCTION
# -------------------------
def main():
    print("🚀 Loading trained model...")
    try:
        # Load model with custom objects
        model = load_model(MODEL_PATH, 
                         custom_objects={'focal_loss_fixed': focal_loss()})
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Make sure the model file exists and the path is correct.")
        return
    
    print("\n📂 Loading and preprocessing test data...")
    
    # Load datasets
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data = load_pickle(POS_DATA_PATH)
    
    # Process emotion data
    X_e = np.array(emotion_data["data"]).reshape(-1, 4, 60)
    X_e = robust_normalize_eeg(X_e)
    y_e = np.array(emotion_data["labels"])
    
    # Process POS data
    X_p = np.array(pos_data["data"]).reshape(-1, 4, 60)
    X_p = robust_normalize_eeg(X_p)
    y_p = np.array(pos_data["labels"]) - 1  # Adjust labels to start from 0
    
    # Add channel dimension
    X_e = X_e[..., np.newaxis]
    X_p = X_p[..., np.newaxis]
    
    # Create test sets (using same split as training for consistency)
    _, X_e_test, _, y_e_test = train_test_split(
        X_e, y_e, test_size=0.2, random_state=RANDOM_STATE, stratify=y_e
    )
    _, X_p_test, _, y_p_test = train_test_split(
        X_p, y_p, test_size=0.2, random_state=RANDOM_STATE, stratify=y_p
    )
    
    print(f"📊 Test set sizes:")
    print(f"  Emotion: {X_e_test.shape[0]} samples")
    print(f"  POS: {X_p_test.shape[0]} samples")
    
    # -------------------------
    # EMOTION TASK EVALUATION
    # -------------------------
    print("\n🎭 Evaluating Emotion Classification...")
    y_pred_emotion, _ = model.predict(X_e_test, verbose=0)
    y_pred_emotion_classes = np.argmax(y_pred_emotion, axis=1)
    
    emotion_accuracy, emotion_cm = create_evaluation_report(
        y_e_test, y_pred_emotion_classes, EMOTION_LABELS, "Emotion"
    )
    
    # -------------------------
    # POS TASK EVALUATION  
    # -------------------------
    print("\n📝 Evaluating POS Classification...")
    _, y_pred_pos = model.predict(X_p_test, verbose=0)
    y_pred_pos_classes = np.argmax(y_pred_pos, axis=1)
    
    pos_accuracy, pos_cm = create_evaluation_report(
        y_p_test, y_pred_pos_classes, POS_LABELS, "POS"
    )
    
    # -------------------------
    # VISUALIZATION
    # -------------------------
    print("\n📈 Creating visualization...")
    
    # Set up the plot
    plt.style.use('default')
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot confusion matrices
    plot_confusion_matrix_with_accuracy(
        emotion_cm, EMOTION_LABELS, 'Emotion Classification', axes[0]
    )
    plot_confusion_matrix_with_accuracy(
        pos_cm, POS_LABELS, 'POS Classification', axes[1]
    )
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('plots/advanced_multi_head_results.png', dpi=300, bbox_inches='tight')
    print("💾 Confusion matrices saved as 'model_evaluation_results.png'")
    
    plt.show()
    
    # -------------------------
    # SUMMARY
    # -------------------------
    print(f"\n{'='*60}")
    print("📋 EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"🎭 Emotion Task Accuracy: {emotion_accuracy:.4f} ({emotion_accuracy*100:.2f}%)")
    print(f"📝 POS Task Accuracy:     {pos_accuracy:.4f} ({pos_accuracy*100:.2f}%)")
    print(f"🔄 Average Accuracy:      {(emotion_accuracy + pos_accuracy)/2:.4f} ({(emotion_accuracy + pos_accuracy)/2*100:.2f}%)")
    print(f"{'='*60}")
    
    # Model info
    print(f"\n🔍 Model Information:")
    print(f"  Total Parameters: {model.count_params():,}")
    print(f"  Model Architecture: {model.name}")
    print(f"  Input Shape: {INPUT_SHAPE}")
    print(f"  Emotion Classes: {NUM_CLASSES_EMOTION}")
    print(f"  POS Classes: {NUM_CLASSES_POS}")
    
    return {
        'emotion_accuracy': emotion_accuracy,
        'pos_accuracy': pos_accuracy,
        'emotion_cm': emotion_cm,
        'pos_cm': pos_cm,
        'model': model
    }

if __name__ == "__main__":
    results = main()
    print("\n✅ Evaluation completed!")