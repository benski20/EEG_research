import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense, MaxPooling2D)
from sklearn.model_selection import train_test_split

model = load_model("eegnet_fft_finetuned_aligned_2_withoutfrozenfirstlayer_and10ofolddata_andnoalignment_withclassweight[1.25:1.25:1]_withaugmentation_noise.h5")  # Load your trained model here

# depthwise_layer = model.get_layer("depthwise_conv2d")  # Adjust name if necessary
# print(f"Depthwise Layer: {depthwise_layer.name}")
# weights = depthwise_layer.get_weights()[0]
# print(weights)  # shape: (channels, 1, 1, depth_multiplier)

# channels = ['Ch1', 'Ch2', 'Ch3', 'Ch4']

# depth_multiplier = weights.shape[-1]
# for i in range(depth_multiplier):
#     channel_weights = weights[:, 0, 0, i]
#     plt.bar(channels, channel_weights)
#     plt.title(f'Spatial (Depthwise) Filter Weight Analysis')
#     plt.ylabel('Weight')
#     plt.ylim(-0.4, 0.4)
#     #if i < 1: 
#         #plt.savefig("plots/eegnet_fft_model_1_regular_eegarch_12500samples_depthwiseweights.png", dpi=300, bbox_inches='tight')
#     plt.show()


with open("eeg_fft_AMY_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["data"])
 # shape: (N, 4, 60) --
X = X.reshape(-1, 4, 60)
print(X.shape) 
y = np.array(data["labels"])    # shape: (N,)

# === Normalize (per trial)
X = (X - X.mean(axis=2, keepdims=True)) / X.std(axis=2, keepdims=True)

# === Reshape for CNN input
X = X[..., np.newaxis]     # shape: (N, 4, 60, 1)

# === Apply static channel reweighting to reduce artifact influence
# TP channels (1 & 2): weight = 1.0
# Frontal channels (3 & 4): weight = 0.6
channel_weights = np.array([0.6, 0.6, 1, 1]).reshape(1, 4, 1, 1)
X = X #* channel_weights  # Apply to all samples but dont use for testing


# === One-hot encode labels
y = to_categorical(y, num_classes=3)

# # === Train/Test Split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

loss, accuracy = model.evaluate(X, y)
print(f"\n✅ Test Accuracy: {accuracy:.4f}")
print(model.summary())

# ############-evaluate a certain trial apart of the test set-##
# import random
# # Select a random trial from the test set
# random_index = random.randint(0, len(X) - 1)
# print(f"Selected Trial Index: {random_index}")
# new_trial = X[random_index]
# new_trial_label = y[random_index]
# new_trial_label = np.argmax(new_trial_label)  # Convert one-hot to class index
# print(f"Trial Label: {new_trial_label}")  # Print the label for the trial


# new_trial = new_trial[np.newaxis, ..., np.newaxis]  # shape: (1, 4, 1000, 1)
#  # Print the one-hot encoded label for the trial

# prediction = model.predict(new_trial)
# predicted_class = np.argmax(prediction)

# print(f"Predicted Class: {predicted_class}")


# #######confusion matrix and classification report for TEST DATA#########
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# # # === Get predictions
y_pred_probs = model.predict(X)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y, axis=1)

# === Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"\n🎯 Test Accuracy: {acc:.4f}")

# === Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred))

# === Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\n🔀 Confusion Matrix:")
print(cm)

# === Plot Confusion Matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# print("Evaluation", model.evaluate(X_test, y_test))
# print("Evaluation Train", model.evaluate(X_train, y_train))  # Evaluate the model on the test set

# # #######confusion matrix and classification report for TRAINING DATA#########
# # Get model predictions on training data
# y_train_pred = model.predict(X_train)
# y_train_pred_classes = np.argmax(y_train_pred, axis=1)

# # Actual labels (if one-hot encoded)
# y_train_true = np.argmax(y_train, axis=1)

# # Accuracy
# train_accuracy = accuracy_score(y_train_true, y_train_pred_classes)
# print("Train Accuracy: {:.4f}".format(train_accuracy))

# # Classification Report
# print("\n📋 Classification Report (Train):")
# print(classification_report(y_train_true, y_train_pred_classes))

# # Confusion Matrix
# print("\n📉 Confusion Matrix (Train):")
# print(confusion_matrix(y_train_true, y_train_pred_classes))

# cm = confusion_matrix(y_train_true, y_train_pred_classes)

# ## === Plot Confusion Matrix
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()


# ########################################## WEIGHT ANALYSIS##########################################
# import matplotlib.pyplot as plt
# import numpy as np

# # Paste your weights array here
# weights = np.array([
#     [[[ 0.05748923,  0.29178855],
#       [-0.05817239, -0.275415  ],
#       [-0.13883075, -0.20503376],
#       [-0.37572566, -0.29699484],
#       [ 0.2709624 , -0.36216348],
#       [-0.03200939,  0.00328988],
#       [ 0.2646482 ,  0.30349126],
#       [ 0.25795683, -0.3304727 ]]],

#     [[[-0.24866225,  0.10866005],
#       [ 0.27884695,  0.32130966],
#       [ 0.15401554,  0.03663451],
#       [ 0.13140364, -0.16777654],
#       [-0.34034193, -0.04246063],
#       [-0.0461994 ,  0.2305472 ],
#       [ 0.17901418,  0.2502046 ],
#       [-0.10127819,  0.1428505 ]]],

#     [[[ 0.22173688,  0.13135357],
#       [-0.14071178,  0.08126811],
#       [ 0.38714898,  0.37316146],
#       [-0.16869365, -0.35925278],
#       [-0.35191143, -0.39765623],
#       [ 0.2250251 , -0.14397159],
#       [ 0.1890568 ,  0.05440388],
#       [ 0.2809163 , -0.04359803]]],

#     [[[-0.05454363,  0.22794233],
#       [ 0.4125063 ,  0.3105638 ],
#       [ 0.3708149 , -0.3813377 ],
#       [-0.2206797 , -0.00454226],
#       [ 0.21226749, -0.14275856],
#       [-0.19025587,  0.21494053],
#       [-0.1634601 ,  0.1443609 ],
#       [-0.40218383,  0.00884878]]]
# ])

# channels = ['Ch1 (TP)', 'Ch2 (TP)', 'Ch3 (Frontal)', 'Ch4 (Frontal)']
# colors = ['tab:blue', 'tab:orange']

# fig, axes = plt.subplots(2, 2, figsize=(12, 6), sharex=True, sharey=True)
# axes = axes.ravel()

# for i in range(4):
#     for d in range(2):  # 2 filters per channel
#         axes[i].plot(range(8), weights[i, 0, :, d], label=f'Filter {d+1}', color=colors[d])
#     axes[i].set_title(f'Channel {i+1} - {channels[i]}')
#     axes[i].set_ylabel('Weight')
#     axes[i].set_xlabel('Kernel Index')
#     axes[i].legend()
#     axes[i].grid(True)

# plt.suptitle('DepthwiseConv2D Weights per EEG Channel', fontsize=14)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
# plt.savefig("plots/eegnet_model_1_depthwiseweightvisualization_artifact.png", dpi=300, bbox_inches='tight')
# plt.show()



# ##########################################ICA ANALYSIS##########################################
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.decomposition import FastICA

# # Simulated or real EEG-like data
# # Example: 100 samples, 4 channels, 1000 timepoints each
# n_samples = 50
# n_channels = 4
# n_timesteps = 1000

# X_flat = X.reshape(n_samples, -1)

# # Apply ICA
# n_components = 10  # Can tune this
# ica = FastICA(n_components=n_components, random_state=42)
# X_ica = ica.fit_transform(X_flat)  # Independent components

# # Retrieve the mixing and unmixing matrices
# mixing_matrix = ica.mixing_  # Shape: (n_components, input_dim)
# unmixing_matrix = ica.components_  # Shape: (n_components, input_dim)

# print(f"ICA Mixing matrix shape: {mixing_matrix.shape}")
# print(f"ICA Components (Unmixing matrix) shape: {unmixing_matrix.shape}")

# # Plotting the ICA components over samples
# plt.figure(figsize=(15, 6))
# for i in range(min(n_components, 5)):
#     plt.plot(X_ica[:, i], label=f'IC {i+1}')
# plt.title('Independent Components across Samples')
# plt.xlabel('Sample Index')
# plt.ylabel('Activation')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # Optional: Visualize components as if reshaped back to (channels x time)
# for i in range(min(n_components, 5)):
#     component_signal = unmixing_matrix[i].reshape(n_channels, n_timesteps)
#     plt.figure(figsize=(12, 4))
#     for ch in range(n_channels):
#         plt.plot(component_signal[ch], label=f'Ch {ch+1}')
#     plt.title(f'ICA Component {i+1} across Channels')
#     plt.xlabel('Time')
#     plt.ylabel('Amplitude')
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# from sklearn.manifold import TSNE
# X_ica_tsne = TSNE(n_components=2, random_state=42).fit_transform(X_ica)
# plt.scatter(X_ica_tsne[:, 0], X_ica_tsne[:, 1], c=y, cmap='viridis', s=10)
# plt.title('ICA + t-SNE colored by true labels')
# plt.show()

# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report

# X_train, X_test, y_train, y_test = train_test_split(X_ica, y, test_size=0.2, random_state=42)

# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

# print(classification_report(y_test, y_pred))

# ########################################T-SNE PLOT########################################
# import numpy as np
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
# import matplotlib.pyplot as plt

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# # --- Step 1: Flatten X if needed ---
# X_flat = X.reshape(X.shape[0], -1)  # (N, 4, 1000, 1) → (N, 4000)

# # --- Step 2: PCA (Optional) before t-SNE for speed ---
# pca = PCA(n_components=50, random_state=42)
# X_pca = pca.fit_transform(X_flat)

# # --- Step 3: t-SNE for 2D projection ---
# X_tsne = TSNE(n_components=2, perplexity=5, random_state=42).fit_transform(X_pca)

# # --- Step 4: KMeans Clustering ---
# kmeans = KMeans(n_clusters=3, random_state=42)
# cluster_labels = kmeans.fit_predict(X_tsne)

# # --- Step 5: Clustering Metrics ---
# sil_score = silhouette_score(X_tsne, cluster_labels)
# dbi = davies_bouldin_score(X_tsne, cluster_labels)
# ch_score = calinski_harabasz_score(X_tsne, cluster_labels)

# print(f"Silhouette Score: {sil_score:.4f}")
# print(f"Davies-Bouldin Index: {dbi:.4f}")
# print(f"Calinski-Harabasz Score: {ch_score:.4f}")

# # --- Step 6: Plot True Labels vs KMeans Clusters ---
# fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# # Plot A: True Labels
# axs[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', s=20)
# axs[0].set_title('True Labels')

# # Plot B: KMeans Clusters
# axs[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='tab10', s=20)
# axs[1].set_title('KMeans Clusters')

# plt.suptitle("t-SNE Projection: True Labels vs. KMeans Clusters", fontsize=14)
# plt.tight_layout()
# plt.show()


# Step 5: Plot t-SNE with cluster assignments
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np

# def plot_clusters_with_hulls(X_2d, labels, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 6))
#     unique_labels = np.unique(labels)
#     colors = plt.cm.get_cmap('tab10', len(unique_labels))

#     for i, label in enumerate(unique_labels):
#         points = X_2d[labels == label]
#         ax.scatter(points[:, 0], points[:, 1], s=30, alpha=0.6, label=f'Cluster {label}', color=colors(i))

#         if len(points) >= 3:  # Convex hull requires at least 3 points
#             hull = ConvexHull(points)
#             hull_points = points[hull.vertices]
#             # Close the polygon by repeating the first point at the end
#             hull_points = np.concatenate([hull_points, hull_points[:1]], axis=0)
#             ax.plot(hull_points[:, 0], hull_points[:, 1], color=colors(i), lw=2, ls='--')

#     ax.legend()
#     ax.set_title('Visible Clusters From t-SNE')
#     plt.savefig("plots/tSNE_clusters_with_hulls.png", dpi=300, bbox_inches='tight')
#     plt.show()

# # Usage example
# plot_clusters_with_hulls(X_tsne, cluster_labels)

# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=cluster_labels, cmap='viridis', s=10)
# plt.title("t-SNE + PCA + KMeans Clustering")
# plt.xlabel("t-SNE Component 1")
# plt.ylabel("t-SNE Component 2")
# plt.colorbar(scatter, label='Cluster')
# plt.tight_layout()
# plt.savefig("plots/tSNE_pca_clustering.png", dpi=300, bbox_inches='tight')
# plt.show()

# Step 6: Calculate and print clustering metrics
# sil_score = silhouette_score(X_tsne, cluster_labels)
# db_score = davies_bouldin_score(X_tsne, cluster_labels)
# ch_score = calinski_harabasz_score(X_tsne, cluster_labels)

# print(f"Silhouette Score: {sil_score:.4f}")
# print(f"Davies-Bouldin Index: {db_score:.4f}")
# print(f"Calinski-Harabasz Score: {ch_score:.4f}")



# ######################################LIME EXPLANATION######################################
# from lime import lime_tabular
# import tensorflow as tf

# # Flatten
# X_train_flat = X_train.reshape(X_train.shape[0], -1)
# X_test_flat = X_test.reshape(X_test.shape[0], -1)

#   # Should be (N, 240)

# # Prediction wrapper
# def predict_fn(input_2d):
#     reshaped = input_2d.reshape(-1, 4, 1000, 1)
#     logits = model.predict(reshaped)
#     return tf.nn.softmax(logits).numpy()  # Ensure probabilities

# # Feature names
# feature_names = [f"ch{ch}_t{t}" for ch in range(4) for t in range(1000)]

# # LIME Explainer
# explainer = lime_tabular.LimeTabularExplainer(
#     training_data=X_train_flat,
#     feature_names=feature_names,
#     class_names=["Class 0", "Class 1", "Class 2"],
#     discretize_continuous=True,
#     mode='classification'
# )

# # Explain one test sample
# sample_index = 0
# sample = X_test_flat[sample_index]
# explanation = explainer.explain_instance(
#     data_row=sample,
#     predict_fn=predict_fn,
#     num_features=60
# )

# # Show explanation
# for feature, weight in explanation.as_list():
#     print(f"{feature}: {weight:.4f}")


# import matplotlib.pyplot as plt

# # Feature names and importance values
# features = [
    
#     "ch2_t681", "ch0_t964", "ch0_t773", "ch0_t933", "ch0_t896", "ch3_t258", "ch0_t634", "ch2_t13", "ch0_t778", "ch3_t703",
#     "ch0_t321", "ch0_t851", "ch0_t674", "ch1_t233", "ch3_t764", "ch3_t192", "ch0_t580", "ch2_t549", "ch0_t364", "ch0_t197",
#     "ch1_t982", "ch1_t378", "ch1_t577", "ch0_t868", "ch1_t640", "ch1_t375", "ch2_t598", "ch2_t852", "ch1_t411", "ch2_t521",
#     "ch1_t480", "ch0_t115", "ch3_t562", "ch0_t654", "ch2_t730", "ch3_t277", "ch2_t256", "ch2_t537", "ch3_t566", "ch1_t290",
#     "ch1_t843", "ch2_t440", "ch1_t144", "ch1_t434", "ch0_t525", "ch2_t559", "ch2_t316", "ch0_t221", "ch1_t657", "ch2_t347",
#     "ch2_t10", "ch1_t28", "ch3_t410", "ch1_t522", "ch0_t32", "ch3_t709", "ch3_t455", "ch3_t276", "ch0_t356", "ch0_t388"
# ]



# importances = [
#     0.0071, 0.0068, 0.0067, -0.0063, -0.0053, -0.0051, -0.0048, -0.0044, 0.0043, -0.0043,
#     -0.0042, 0.0042, 0.0039, 0.0036, -0.0035, -0.0034, 0.0034, -0.0032, 0.0031, -0.0031,
#     0.0030, -0.0030, -0.0028, -0.0028, 0.0028, 0.0028, -0.0028, -0.0027, -0.0027, -0.0027,
#     0.0027, -0.0027, -0.0025, -0.0023, 0.0022, 0.0022, 0.0021, -0.0020, -0.0019, 0.0019,
#     -0.0019, -0.0018, 0.0017, 0.0016, 0.0016, -0.0014, 0.0013, -0.0013, -0.0012, 0.0011,
#     0.0010, -0.0009, -0.0009, 0.0009, -0.0008, 0.0008, 0.0008, 0.0005, -0.0004, 0.0002
# ]



# # Sort features by absolute importance for better visualization
# sorted_indices = sorted(range(len(importances)), key=lambda i: abs(importances[i]), reverse=True)
# sorted_features = [features[i] for i in sorted_indices]
# sorted_importances = [importances[i] for i in sorted_indices]

# # Plot
# plt.figure(figsize=(16, 8))
# bars = plt.bar(range(len(sorted_importances)), sorted_importances, color='teal')
# plt.xticks(range(len(sorted_features)), sorted_features, rotation=45, ha='right')
# plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
# plt.title("Top (60) Feature Importance Scores For Model Weighted EEGNet Model (Weights: [1.0, 1.0, 0.6, 0.6])")
# plt.ylabel("Importance")
# plt.fontsize = 4
# plt.tight_layout()
# plt.grid(axis='y', linestyle='--', alpha=0.5)

# # Annotate bars
# for i, val in enumerate(sorted_importances):
#     plt.text(i, val + 0.01 * (1 if val > 0 else -1), f"{val:.3f}", ha='center', va='bottom' if val > 0 else 'top', fontsize=4)
# plt.savefig("plots/eegnet_model_1_averagepooling_50epochs_channelweighted[1166]_feature_importance.png", dpi=300, bbox_inches='tight')
# plt.show()


# ###############################################LIME PLOT (AVERAGE CONTRIBUTION PER CHANNEL)###############################################
# import re
# from collections import defaultdict
# import numpy as np
# import matplotlib.pyplot as plt

# # # === Step 1: Raw Input (Feature : Importance)
# raw_input = """
# ch0_t965 > 0.53: -0.0071
# ch0_t734 <= -0.89: 0.0058
# 0.09 < ch1_t621 <= 0.93: -0.0056
# ch0_t63 > 0.69: -0.0054
# ch1_t738 <= -0.90: -0.0052
# ch0_t391 <= -0.49: 0.0050
# ch2_t181 <= -0.31: 0.0049
# -0.00 < ch3_t956 <= 0.28: 0.0049
# ch0_t71 <= -0.80: 0.0048
# ch2_t564 > 0.25: 0.0046
# ch2_t386 > 0.36: -0.0043
# -0.60 < ch0_t962 <= -0.11: 0.0041
# ch2_t778 <= -0.29: -0.0041
# ch0_t636 > 0.77: 0.0040
# 0.06 < ch3_t324 <= 0.24: 0.0037
# ch3_t402 > 0.19: -0.0037
# -0.02 < ch0_t895 <= 0.99: -0.0036
# ch2_t689 <= -0.23: 0.0035
# -0.25 < ch0_t650 <= 0.53: -0.0034
# ch0_t939 > 0.90: 0.0034
# ch0_t137 <= -1.14: -0.0032
# -0.00 < ch2_t347 <= 0.28: -0.0032
# -0.81 < ch1_t57 <= -0.27: -0.0032
# ch0_t732 > 0.52: -0.0031
# ch2_t874 > 0.33: -0.0031
# ch2_t298 <= -0.33: 0.0030
# -0.21 < ch3_t949 <= 0.04: -0.0030
# ch0_t194 <= -0.55: -0.0029
# 0.09 < ch1_t542 <= 0.84: -0.0029
# ch0_t929 > 0.81: 0.0029
# 0.10 < ch1_t518 <= 0.83: 0.0027
# -0.15 < ch3_t66 <= 0.04: -0.0027
# -0.84 < ch0_t492 <= -0.35: 0.0027
# 0.24 < ch1_t591 <= 1.00: -0.0027
# ch0_t148 > 0.94: 0.0025
# 0.12 < ch0_t607 <= 0.96: -0.0025
# -0.03 < ch2_t858 <= 0.47: 0.0024
# 0.01 < ch3_t883 <= 0.27: 0.0024
# -0.24 < ch2_t948 <= 0.02: 0.0023
# ch0_t242 > 0.73: 0.0022
# -1.24 < ch1_t753 <= -0.51: -0.0021
# ch3_t165 > 0.33: 0.0020
# ch1_t706 > 0.76: -0.0018
# -0.23 < ch2_t439 <= -0.03: 0.0017
# ch0_t291 > 1.02: 0.0016
# ch1_t345 > 0.76: -0.0016
# -1.05 < ch1_t396 <= -0.27: 0.0015
# -0.28 < ch3_t59 <= -0.05: 0.0013
# ch0_t31 <= -0.79: -0.0013
# ch3_t602 > 0.14: 0.0012
# ch1_t905 <= -0.45: -0.0010
# -0.04 < ch1_t117 <= 0.75: 0.0010
# -0.81 < ch1_t850 <= -0.07: 0.0008
# -0.04 < ch3_t964 <= 0.22: -0.0007
# -0.82 < ch1_t707 <= 0.01: 0.0007
# ch1_t587 <= -0.67: 0.0006
# ch1_t320 > 0.62: -0.0002
# ch3_t877 <= -0.14: -0.0002
# -0.11 < ch3_t896 <= 0.04: 0.0001
# 0.12 < ch1_t759 <= 1.04: 0.0001
# """

# === Step 2: Extract feature -> importance
# import re
# pattern = re.compile(r"(ch\d+_t\d+)[^:]*:\s*(-?\d+\.\d+)")
# matches = pattern.findall(raw_input)

# features = [m[0] for m in matches]
# importances = [float(m[1]) for m in matches]

# # === Step 3: Average contribution per channel
# channel_sums = defaultdict(list)
# for feat, imp in zip(features, importances):
#     ch = feat.split("_")[0]  # ch0, ch1, etc.
#     channel_sums[ch].append(imp)

# channel_avg = {ch: np.mean(vals) for ch, vals in channel_sums.items()}

# # === Step 4: Plot
# plt.figure(figsize=(7, 4))
# channels = sorted(channel_avg.keys(), key=lambda x: int(x[2:]))
# avg_vals = [channel_avg[ch] for ch in channels]

# plt.bar(channels, avg_vals, color='skyblue')
# plt.title("Average LIME Contribution per Channel For Model Unweighted EEGNet Model")
# plt.xlabel("Channel")
# plt.ylabel("Average Contribution")
# plt.axhline(0, color='gray', linestyle='--', linewidth=1)

# plt.tight_layout()
# #plt.savefig("plots/eegnet_model_1_weightedchannels[1166]_averagechannel_importance_WITHOUTWEIGHTEDINPUTNOTESIMILARITIESBUTDONOTINCLUDE.png", dpi=300, bbox_inches='tight')
# plt.show()
