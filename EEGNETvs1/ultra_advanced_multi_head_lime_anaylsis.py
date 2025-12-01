# lime_channel_analysis.py - LIME analysis for EEG channel importance
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

# -------------------------
# CONFIG
# -------------------------
EMOTION_DATA_PATH = "eeg_fft_dataset.pkl"
POS_DATA_PATH = "eeg_fft_nounvsverb_JASON_dataset.pkl"
MODEL_PATH = "best_balanced_fusion_model.h5"

NUM_CLASSES_EMOTION = 3
NUM_CLASSES_POS = 2
RANDOM_STATE = 42

# Task-specific channel names
CHANNEL_NAMES_EMOTION = ['Emotion Ch1 (TP7)', 'Emotion Ch2 (TP8)', 'Emotion Ch3 (F7)', 'Emotion Ch4 (F8)']
CHANNEL_NAMES_POS = ['POS Ch1 (T7)', 'POS Ch2 (T8)', 'POS Ch3 (F7)', 'POS Ch4 (F8)']

# You can customize these based on your actual electrode placements:
# For emotion task - example emotional processing regions:
# CHANNEL_NAMES_EMOTION = ['Fp1 (Prefrontal)', 'F4 (Right Frontal)', 'T3 (Left Temporal)', 'P4 (Right Parietal)']

# For POS task - example language processing regions:
# CHANNEL_NAMES_POS = ['F7 (Broca)', 'T5 (Wernicke)', 'P3 (Angular)', 'C3 (Motor)']

def get_channel_names(task):
    """Get appropriate channel names for the specified task"""
    if task == 'emotion':
        return CHANNEL_NAMES_EMOTION
    elif task == 'pos':
        return CHANNEL_NAMES_POS
    else:
        raise ValueError(f"Unknown task: {task}")

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
# DATA PREPROCESSING (same as training)
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
# LIME ANALYSIS FUNCTIONS
# -------------------------
def prepare_data_for_lime(X):
    """
    Convert EEG data to format suitable for LIME
    X shape: (samples, channels, time_points, 1)
    Returns: (samples, features) where features = channels * time_points
    """
    # Remove the last dimension and flatten to (samples, channels * time_points)
    X_lime = X.squeeze(-1)  # Remove channel dimension -> (samples, channels, time_points)
    X_lime = X_lime.reshape(X_lime.shape[0], -1)  # Flatten -> (samples, channels * time_points)
    return X_lime

def create_channel_feature_map(n_channels, n_timepoints):
    """
    Create mapping from flattened features back to channels
    Returns: array where each element indicates which channel that feature belongs to
    """
    channel_map = []
    for ch in range(n_channels):
        channel_map.extend([ch] * n_timepoints)
    return np.array(channel_map)

class EEGModelWrapper:
    """Wrapper to make model compatible with LIME"""
    def __init__(self, model, task='emotion'):
        self.model = model
        self.task = task  # 'emotion' or 'pos'
        
    def predict_proba(self, X_flat):
        # Reshape from (samples, channels * time_points) back to model input format
        n_samples = X_flat.shape[0]
        X_reshaped = X_flat.reshape(n_samples, 4, 60, 1)  # Back to original shape
        
        # Get predictions for both tasks
        emotion_pred, pos_pred = self.model.predict(X_reshaped, verbose=0)
        
        if self.task == 'emotion':
            return emotion_pred
        else:
            return pos_pred

def compute_lime_importance(model, X_sample, X_background, task='emotion', n_samples=5000):
    """
    Compute LIME importance for a single sample
    """
    # Prepare data for LIME
    X_sample_lime = prepare_data_for_lime(X_sample.reshape(1, 4, 60, 1))
    X_background_lime = prepare_data_for_lime(X_background)
    
    # Create model wrapper
    model_wrapper = EEGModelWrapper(model, task=task)
    
    # Get task-specific channel names for feature naming
    channel_names = get_channel_names(task)
    
    # Create LIME explainer with task-specific feature names
    explainer = lime_tabular.LimeTabularExplainer(
        X_background_lime,
        mode='classification',
        feature_names=[f'{channel_names[ch]}_t{t}' for ch in range(4) for t in range(60)],
        class_names=['Class_0', 'Class_1', 'Class_2'] if task == 'emotion' else ['Noun', 'Verb'],
        discretize_continuous=False
    )
    
    # Get explanation
    explanation = explainer.explain_instance(
        X_sample_lime[0], 
        model_wrapper.predict_proba,
        num_features=240,  # All features (4 channels * 60 time points)
        num_samples=n_samples
    )
    
    return explanation

def analyze_channel_importance(model, X_data, sample_weights, task='emotion', n_samples_to_analyze=50):
    """
    Analyze channel importance across multiple samples
    """
    print(f"🧠 Analyzing {task} task channel importance...")
    
    # Get task-specific channel names and class names
    channel_names = get_channel_names(task)
    
    # Filter data for the specific task
    if task == 'emotion':
        mask = sample_weights > 0
        task_data = X_data[mask]
        task_name = "Emotion Classification"
        class_names = ["Negative", "Neutral", "Positive"]
    else:
        mask = sample_weights > 0
        task_data = X_data[mask]
        task_name = "POS Classification"
        class_names = ["Noun", "Verb"]
    
    if len(task_data) == 0:
        print(f"⚠️ No data found for {task} task!")
        return None
    
    print(f"📊 Found {len(task_data)} samples for {task} task")
    print(f"📊 Using channel names: {channel_names}")
    
    # Randomly sample data points to analyze
    n_analyze = min(n_samples_to_analyze, len(task_data))
    sample_indices = np.random.choice(len(task_data), n_analyze, replace=False)
    
    # Create background dataset (larger sample for LIME)
    n_background = min(1000, len(task_data))
    background_indices = np.random.choice(len(task_data), n_background, replace=False)
    X_background = task_data[background_indices]
    
    # Initialize channel importance storage
    channel_importance_per_sample = []
    channel_importance_per_class = {i: [] for i in range(len(class_names))}
    
    # Get model predictions for class-specific analysis
    predictions = model.predict(task_data[sample_indices], verbose=0)
    if task == 'emotion':
        pred_classes = np.argmax(predictions[0], axis=1)
    else:
        pred_classes = np.argmax(predictions[1], axis=1)
    
    print(f"🔍 Analyzing {n_analyze} samples with LIME...")
    
    for i, sample_idx in enumerate(sample_indices):
        if i % 10 == 0:
            print(f"   Progress: {i}/{n_analyze}")
        
        try:
            # Get LIME explanation for this sample
            explanation = compute_lime_importance(
                model, 
                task_data[sample_idx], 
                X_background, 
                task=task,
                n_samples=1000  # Reduced for speed
            )
            
            # Extract feature importance
            feature_importance = explanation.as_map()[pred_classes[i]]
            importance_dict = dict(feature_importance)
            
            # Map feature importance to channels
            channel_map = create_channel_feature_map(4, 60)
            channel_importance = np.zeros(4)
            
            for feature_idx, importance in importance_dict.items():
                channel = channel_map[feature_idx]
                channel_importance[channel] += abs(importance)  # Use absolute importance
            
            # Normalize by number of time points per channel
            channel_importance = channel_importance / 60
            
            channel_importance_per_sample.append(channel_importance)
            channel_importance_per_class[pred_classes[i]].append(channel_importance)
            
        except Exception as e:
            print(f"   ⚠️ Error analyzing sample {i}: {e}")
            continue
    
    if len(channel_importance_per_sample) == 0:
        print("❌ No successful LIME analyses!")
        return None
    
    # Compute average importance
    avg_channel_importance = np.mean(channel_importance_per_sample, axis=0)
    std_channel_importance = np.std(channel_importance_per_sample, axis=0)
    
    # Compute class-specific importance
    class_channel_importance = {}
    for class_idx, class_name in enumerate(class_names):
        if len(channel_importance_per_class[class_idx]) > 0:
            class_channel_importance[class_name] = np.mean(channel_importance_per_class[class_idx], axis=0)
        else:
            class_channel_importance[class_name] = np.zeros(4)
    
    results = {
        'task': task_name,
        'task_key': task,
        'channel_names': channel_names,
        'avg_importance': avg_channel_importance,
        'std_importance': std_channel_importance,
        'class_importance': class_channel_importance,
        'n_samples_analyzed': len(channel_importance_per_sample),
        'class_names': class_names
    }
    
    print(f"✅ {task_name} analysis complete!")
    print(f"📊 Average channel importance: {avg_channel_importance}")
    
    return results

def plot_channel_importance(results_emotion, results_pos, save_path=None):
    """
    Plot channel importance for both tasks with task-specific channel names
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('EEG Channel Importance Analysis via LIME\n(Task-Specific Channel Configurations)', 
                 fontsize=16, fontweight='bold')
    
    # Color schemes
    emotion_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    pos_colors = ['#A8E6CF', '#DCEDC8', '#C8E6C9', '#B2DFDB']
    
    # ========================
    # EMOTION TASK PLOTS
    # ========================
    if results_emotion is not None:
        channel_names_emotion = results_emotion['channel_names']
        
        # Overall emotion importance
        ax1 = axes[0, 0]
        bars1 = ax1.bar(range(len(channel_names_emotion)), results_emotion['avg_importance'], 
                       yerr=results_emotion['std_importance'], 
                       capsize=5, color=emotion_colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax1.set_title('Emotion Classification - Overall Channel Importance', fontweight='bold')
        ax1.set_ylabel('Average LIME Importance')
        ax1.set_xticks(range(len(channel_names_emotion)))
        ax1.set_xticklabels(channel_names_emotion, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars1, results_emotion['avg_importance']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Class-specific emotion importance
        ax2 = axes[0, 1]
        class_names = results_emotion['class_names']
        x_pos = np.arange(len(channel_names_emotion))
        width = 0.25
        
        for i, class_name in enumerate(class_names):
            importance = results_emotion['class_importance'][class_name]
            ax2.bar(x_pos + i*width, importance, width, 
                   label=class_name, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax2.set_title('Emotion Classification - Per-Class Channel Importance', fontweight='bold')
        ax2.set_ylabel('Average LIME Importance')
        ax2.set_xticks(x_pos + width)
        ax2.set_xticklabels(channel_names_emotion, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
    
    # ========================
    # POS TASK PLOTS
    # ========================
    if results_pos is not None:
        channel_names_pos = results_pos['channel_names']
        
        # Overall POS importance
        ax3 = axes[1, 0]
        bars3 = ax3.bar(range(len(channel_names_pos)), results_pos['avg_importance'], 
                       yerr=results_pos['std_importance'], 
                       capsize=5, color=pos_colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax3.set_title('POS Classification - Overall Channel Importance', fontweight='bold')
        ax3.set_ylabel('Average LIME Importance')
        ax3.set_xticks(range(len(channel_names_pos)))
        ax3.set_xticklabels(channel_names_pos, rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars3, results_pos['avg_importance']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Class-specific POS importance
        ax4 = axes[1, 1]
        class_names = results_pos['class_names']
        x_pos = np.arange(len(channel_names_pos))
        width = 0.35
        
        for i, class_name in enumerate(class_names):
            importance = results_pos['class_importance'][class_name]
            ax4.bar(x_pos + i*width, importance, width, 
                   label=class_name, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax4.set_title('POS Classification - Per-Class Channel Importance', fontweight='bold')
        ax4.set_ylabel('Average LIME Importance')
        ax4.set_xticks(x_pos + width/2)
        ax4.set_xticklabels(channel_names_pos, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    #fig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========================
    # SUMMARY COMPARISON PLOT
    # ========================
    if results_emotion is not None and results_pos is not None:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Emotion task subplot
        ax_left.bar(range(len(results_emotion['channel_names'])), results_emotion['avg_importance'], 
                   color='#FF6B6B', alpha=0.8, edgecolor='black')
        ax_left.set_title('Emotion Classification\nChannel Importance', fontweight='bold')
        ax_left.set_ylabel('Average LIME Importance')
        ax_left.set_xticks(range(len(results_emotion['channel_names'])))
        ax_left.set_xticklabels(results_emotion['channel_names'], rotation=45, ha='right')
        ax_left.grid(axis='y', alpha=0.3)
        
        # POS task subplot
        ax_right.bar(range(len(results_pos['channel_names'])), results_pos['avg_importance'], 
                    color='#4ECDC4', alpha=0.8, edgecolor='black')
        ax_right.set_title('POS Classification\nChannel Importance', fontweight='bold')
        ax_right.set_ylabel('Average LIME Importance')
        ax_right.set_xticks(range(len(results_pos['channel_names'])))
        ax_right.set_xticklabels(results_pos['channel_names'], rotation=45, ha='right')
        ax_right.grid(axis='y', alpha=0.3)
        
        plt.suptitle('Channel Importance Comparison: Emotion vs POS Classification', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        #plt.savefig('plots/third_weighted_ultra_advanced_multi_head_channel_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

def print_analysis_summary(results_emotion, results_pos):
    """Print summary of LIME analysis results with task-specific channel names"""
    print("\n" + "="*80)
    print("🧠 LIME CHANNEL IMPORTANCE ANALYSIS SUMMARY")
    print("="*80)
    
    if results_emotion is not None:
        print(f"\n🎭 EMOTION CLASSIFICATION RESULTS:")
        print(f"   Samples analyzed: {results_emotion['n_samples_analyzed']}")
        print(f"   Channel configuration: {results_emotion['channel_names']}")
        print(f"   Channel ranking (most to least important):")
        
        importance = results_emotion['avg_importance']
        channel_names = results_emotion['channel_names']
        sorted_indices = np.argsort(importance)[::-1]
        
        for i, ch_idx in enumerate(sorted_indices):
            print(f"   {i+1}. {channel_names[ch_idx]}: {importance[ch_idx]:.4f}")
    
    if results_pos is not None:
        print(f"\n📝 POS CLASSIFICATION RESULTS:")
        print(f"   Samples analyzed: {results_pos['n_samples_analyzed']}")
        print(f"   Channel configuration: {results_pos['channel_names']}")
        print(f"   Channel ranking (most to least important):")
        
        importance = results_pos['avg_importance']
        channel_names = results_pos['channel_names']
        sorted_indices = np.argsort(importance)[::-1]
        
        for i, ch_idx in enumerate(sorted_indices):
            print(f"   {i+1}. {channel_names[ch_idx]}: {importance[ch_idx]:.4f}")
    
    print("\n💡 INTERPRETATION:")
    print("   - Higher values indicate greater channel importance")
    print("   - LIME importance shows which brain regions the model relies on most")
    print("   - Different channel configurations reflect task-specific neural processing")
    print("   - Emotion channels may focus on limbic and prefrontal regions")
    print("   - POS channels may focus on language-specific cortical areas")
    print("="*80)

# -------------------------
# MAIN EXECUTION
# -------------------------
def main():
    print("🚀 Starting LIME Channel Importance Analysis...")
    print("📦 Installing lime if not available...")
    
    try:
        import lime
    except ImportError:
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "lime"])
        import lime
    
    # Display channel configurations
    print("\n📡 CHANNEL CONFIGURATIONS:")
    print(f"   Emotion Task Channels: {CHANNEL_NAMES_EMOTION}")
    print(f"   POS Task Channels: {CHANNEL_NAMES_POS}")
    
    # Load data (same preprocessing as training)
    print("\n📂 Loading datasets...")
    emotion_data = load_pickle(EMOTION_DATA_PATH)
    pos_data = load_pickle(POS_DATA_PATH)
    
    # Process emotion data
    X_e = np.array(emotion_data["data"]).reshape(-1, 4, 60)
    y_e = np.array(emotion_data["labels"])
    X_e_improved, y_e_improved = apply_conservative_emotion_improvements(X_e, y_e)
    
    # Process POS data
    X_p = np.array(pos_data["data"]).reshape(-1, 4, 60)
    X_p = robust_normalize_eeg(X_p)
    y_p = np.array(pos_data["labels"]) - 1
    
    # Apply SMOTE to POS if needed
    X_p_balanced, y_p_balanced = X_p, y_p
    pos_counts = np.bincount(y_p)
    if len(pos_counts) > 1 and np.max(pos_counts) > np.min(pos_counts) * 1.5:
        original_shape = X_p.shape
        X_p_flat = X_p.reshape(X_p.shape[0], -1)
        smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
        X_p_flat, y_p_balanced = smote.fit_resample(X_p_flat, y_p)
        X_p_balanced = X_p_flat.reshape(-1, *original_shape[1:])
    
    # Add channel dimension and combine
    X_e_improved = X_e_improved[..., np.newaxis]
    X_p_balanced = X_p_balanced[..., np.newaxis]
    X_combined = np.concatenate([X_e_improved, X_p_balanced], axis=0)
    #channel_weights = np.array([1, 1, 0.75, 0.75]).reshape(1, 4, 1)
    #X_e = X_e * channel_weights
    # Sample weights
    sw_emotion = np.concatenate([np.ones(len(X_e_improved)), np.zeros(len(X_p_balanced))], axis=0)
    sw_pos = np.concatenate([np.zeros(len(X_e_improved)), np.ones(len(X_p_balanced))], axis=0)
    
    # Create validation split
    train_idx, val_idx = create_stratified_split(
        X_combined, y_e_improved, y_p_balanced, test_size=0.2, random_state=RANDOM_STATE
    )
    
    X_val = X_combined[val_idx]
    swe_val = sw_emotion[val_idx]
    swp_val = sw_pos[val_idx]
    
    print(f"📊 Validation set: {len(X_val)} samples")
    print(f"📊 Emotion samples: {np.sum(swe_val > 0)}")
    print(f"📊 POS samples: {np.sum(swp_val > 0)}")
    
    # Load model
    print("🔄 Loading trained model...")
    model = load_model(MODEL_PATH, custom_objects={'focal_loss_fixed': balanced_focal_loss()})
    
    # Perform LIME analysis
    print("🧠 Performing LIME analysis...")
    
    # Analyze emotion task
    results_emotion = analyze_channel_importance(
        model, X_val, swe_val, task='emotion', n_samples_to_analyze=30
    )
    
    # Analyze POS task
    results_pos = analyze_channel_importance(
        model, X_val, swp_val, task='pos', n_samples_to_analyze=30
    )
    
    # Create visualizations
    print("📊 Creating visualizations...")
    plot_channel_importance(results_emotion, results_pos)
    
    # Print summary
    print_analysis_summary(results_emotion, results_pos)
    
    print("✅ LIME analysis complete!")
    print("📁 Saved: channel_importance_analysis.png")
    print("📁 Saved: channel_importance_comparison.png")
    
    return results_emotion, results_pos

if __name__ == "__main__":
    emotion_results, pos_results = main()