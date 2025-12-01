from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
import pickle
import numpy as np
import os
from sklearn.model_selection import train_test_split
with open("eeg_raw_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["X"])    # shape: (N, 4, 1000)
y = np.array(data["y"])    # shape: (N,)
# X = X.reshape(-1, 4, 60)    # shape: (N,)

# === Normalize (per trial)
X = (X - X.mean(axis=2, keepdims=True)) / X.std(axis=2, keepdims=True)

# === Reshape for CNN input
X = X[..., np.newaxis]     # shape: (N, 4, 1000, 1)

# === Apply static channel reweighting to reduce artifact influence
# TP channels (1 & 2): weight = 1.0
# Frontal channels (3 & 4): weight = 0.6
channel_weights = np.array([1, 0.6, 0.6, 1]).reshape(1, 4, 1, 1) ##not needed for attention model
X = X #* channel_weights  # Apply to all samples but dont use for testing

# y = y - 1
# === One-hot encode labels
y = to_categorical(y, num_classes=3)

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
print(len(X_train), len(X_test))
def evaluate_attention_model(model_path, X_train, y_train, X_test, y_test, n_runs=100):
    print(f"Evaluating {model_path} for {n_runs} runs...")

    model = load_model(model_path)
    print("Model loaded successfully.")
    
    X_train_w = X_train
    X_test_w = X_test

    train_accs, test_accs = [], []
    f1s, recalls = [], []
    cms = []

    for _ in range(n_runs):
        # Predict on train and test sets
        idx = np.random.choice(len(X_test_w), int(0.4 * len(X_test_w)), replace=True)
        X_test_sample = X_test[idx]
        y_test_sample = y_test[idx]
        print(idx)

        train_pred_probs = model.predict(X_train, verbose=0)  # Train set not sampled
        train_preds = np.argmax(train_pred_probs, axis=1)
        train_acc = accuracy_score(np.argmax(y_train, axis=1), train_preds)
        train_accs.append(train_acc)

        test_pred_probs = model.predict(X_test_sample, verbose=0)
        test_preds = np.argmax(test_pred_probs, axis=1)

        test_acc = accuracy_score(np.argmax(y_test_sample, axis=1), test_preds)
        recall = recall_score(np.argmax(y_test_sample, axis=1), test_preds, average='macro')
        f1 = f1_score(np.argmax(y_test_sample, axis=1), test_preds, average='macro')
        cm = confusion_matrix(np.argmax(y_test_sample, axis=1), test_preds, labels=[0, 1, 2])

        test_accs.append(test_acc)
        recalls.append(recall)
        f1s.append(f1)
        cms.append(cm)
    
        print(test_accs)

    avg_train_acc = np.mean(train_accs)
    avg_test_acc = np.mean(test_accs)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1s)
    avg_cm = np.mean(cms, axis=0)
    

    return {
        "train_accuracy": avg_train_acc,
        "test_accuracy": avg_test_acc,
        "recall": avg_recall,
        "f1_score": avg_f1,
        "avg_confusion_matrix": avg_cm
    }

def save_results(results, model_name):
    base_name = os.path.splitext(os.path.basename(model_name))[0]
    metrics_file = os.path.join(output_dir, f"{base_name}_metrics.txt")
    cm_file = os.path.join(output_dir, f"{base_name}_confusion_matrix.npy")

    with open(metrics_file, "w") as f:
        for k, v in results.items():
            if k != "avg_confusion_matrix":
                f.write(f"{k}: {v:.4f}\n")

    np.save(cm_file, results["avg_confusion_matrix"])

    print(f"Saved metrics to {metrics_file}")
    print(f"Saved confusion matrix to {cm_file}")

# === Run evaluation for all attention models ===
attention_model_files = [
    "eegnet_model_1.h5"
]

output_dir = "original_model_evaluation_results_corrected"
os.makedirs(output_dir, exist_ok=True)

for model_file in attention_model_files:
    results = evaluate_attention_model(model_file, X_train, y_train, X_test, y_test, n_runs=100)
    save_results(results, model_file)



