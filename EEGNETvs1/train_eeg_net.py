# train_eegnet.py
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (Input, Conv2D, DepthwiseConv2D, SeparableConv2D,
                                     BatchNormalization, Activation, AveragePooling2D,
                                     Dropout, Flatten, Dense, MaxPooling2D)
from sklearn.model_selection import train_test_split

# === EEGNet Model ===
def EEGNet(nb_classes, Chans=4, Samples=1000, dropoutRate=0.5, kernLength=64, F1=8, D=2, F2=16, norm_rate=0.25):
    input_main = Input(shape=(Chans, Samples, 1))

    x = Conv2D(F1, (1, kernLength), padding='same', use_bias=False)(input_main)
    x = BatchNormalization()(x)

    x = DepthwiseConv2D((Chans, 1), use_bias=False,
                        depth_multiplier=D)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 4))(x)
    x = Dropout(dropoutRate)(x)

    x = SeparableConv2D(F2, (1, 16), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = AveragePooling2D((1, 8))(x)
    x = Dropout(dropoutRate)(x)

    x = Flatten(name='flatten')(x)
    x = Dense(nb_classes, activation='softmax')(x)

    return Model(inputs=input_main, outputs=x)

# === Load Data ===
with open("eeg_raw_dataset.pkl", "rb") as f:
    data = pickle.load(f)

X = np.array(data["X"])    # shape: (50, 4, 1000) 
y = np.array(data["y"])    # shape: (50,)

# === Normalize (per trial)
X = (X - X.mean(axis=2, keepdims=True)) / X.std(axis=2, keepdims=True)

# === Reshape for CNN input
X = X[..., np.newaxis]     # shape: (50, 4, 1000, 1)

# === Apply static channel reweighting to reduce artifact influence
# TP channels (1 & 2): weight = 1.0
# Frontal channels (3 & 4): weight = 0.6
channel_weights = np.array([1, 1, 0.6, 0.6]).reshape(1, 4, 1, 1)
X = X # channel_weights  # Apply to all samples but dont use for testing


# === One-hot encode labels
y = to_categorical(y, num_classes=3)

# === Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# === Build and Train EEGNet
# model = EEGNet(nb_classes=3, Chans=4, Samples=1000)
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# model.fit(X_train, y_train, batch_size=8, epochs=50, validation_data=(X_test, y_test), verbose=1)

# === Save model (optional)
import time
# model.save(f"eegnet_model_1_averagepooling_50epochs_channelweighted[5501]{time.strftime('%Y%m%d-%H%M')}.h5")
print("\n✅ Model trained and saved")
model = load_model("eegnet_model_1_averagepooling_50epochs_channelweighted[1166]20250731-2243.h5")  # Load your trained model here
# loss, accuracy = model.evaluate(X_test, y_test)
# print(f"\n✅ Test Accuracy: {accuracy:.4f}")
# print(model.summary())


# ############-evaluate a certain trial apart of the test set-##
# import random
# # Select a random trial from the test set
# random_index = random.randint(0, len(X_test) - 1)
# print(f"Selected Trial Index: {random_index}")
# new_trial = X_test[random_index]
# new_trial_label = y_test[random_index]
# new_trial_label = np.argmax(new_trial_label)  # Convert one-hot to class index
# print(f"Trial Label: {new_trial_label}")  # Print the label for the trial


# new_trial = new_trial[np.newaxis, ..., np.newaxis]  # shape: (1, 4, 1000, 1)
#  # Print the one-hot encoded label for the trial

# prediction = model.predict(new_trial)
# predicted_class = np.argmax(prediction)

# print(f"Predicted Class: {predicted_class}")


# #######confusion matrix and classification report for TEST DATA#########
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# import seaborn as sns
# import matplotlib.pyplot as plt

# # # # === Get predictions
# y_pred_probs = model.predict(X_test)
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.argmax(y_test, axis=1)

# # === Accuracy
# acc = accuracy_score(y_true, y_pred)
# print(f"\n🎯 Test Accuracy: {acc:.4f}")

# # === Classification Report
# print("\n📋 Classification Report:")
# print(classification_report(y_true, y_pred))

# # === Confusion Matrix
# cm = confusion_matrix(y_true, y_pred)
# print("\n🔀 Confusion Matrix:")
# print(cm)

# # === Plot Confusion Matrix
# plt.figure(figsize=(6, 5))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.show()

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


# ############################LIME Visualization#############################
# ##=== Flatten input for LIME: (N, 4, 1000, 1) -> (N, 4000)
# from lime import lime_tabular

# X_train_flat = X_train.reshape(X_train.shape[0], -1)
# X_test_flat = X_test.reshape(X_test.shape[0], -1)

# # === Define LIME prediction wrapper
# def predict_fn(input_2d):
#     # Reshape back to (N, 4, 1000, 1)
#     reshaped = input_2d.reshape(-1, 4, 1000, 1)
#     return model.predict(reshaped)

# # === Define feature names (optional but helpful for interpretability)
# feature_names = [f"ch{ch}_t{t}" for ch in range(4) for t in range(1000)]

# # === Initialize LIME Tabular Explainer
# explainer = lime_tabular.LimeTabularExplainer(
#     training_data=X_train_flat,
#     feature_names=feature_names,
#     class_names=["Class 0", "Class 1", "Class 2"],
#     discretize_continuous=True,
#     mode='classification'
# )

# # === Pick one test sample to explain
# sample_index = 0
# sample = X_test_flat[sample_index]

# # === Run explanation
# explanation = explainer.explain_instance(
#     data_row=sample,
#     predict_fn=predict_fn,
#     num_features=10  # Top 10 contributing features
# )

# print(f"\n🔍 Explanation for Sample {explanation}:")

# # === Optionally, print top features to console
# print("\n🔍 Top contributing features:")
# for feature, weight in explanation.as_list():
#     print(f"{feature}: {weight:.4f}")

# import matplotlib.pyplot as plt

# # Your LIME explanation as a list of tuples (feature_name, weight)
# lime_features = [
#     ("ch2_t707 > 0.31", -0.0085),
#     ("ch3_t590 <= -0.18", -0.0066),
#     ("ch3_t350 <= -0.35", 0.0052),
#     ("-0.17 < ch2_t741 <= 0.26", 0.0044),
#     ("-0.71 < ch1_t225 <= 0.44", 0.0033),
#     ("ch2_t491 <= -0.36", 0.0022),
#     ("-0.34 < ch2_t52 <= 0.01", -0.0019),
#     ("-0.28 < ch2_t530 <= 0.12", -0.0018),
#     ("ch0_t931 > 0.92", -0.0014),
#     ("ch1_t569 > 0.58", -0.0014),
# ]

# # Separate names and values
# features, weights = zip(*lime_features)

# # Plot
# plt.figure(figsize=(10, 5))
# colors = ['green' if w > 0 else 'red' for w in weights]
# plt.barh(features, weights, color=colors)
# plt.xlabel('Contribution to Prediction')
# plt.title('Top LIME Feature Contributions')
# plt.gca().invert_yaxis()  # Most important on top
# plt.tight_layout()
# plt.show()



# ###############LIME FOR MULTIPLE SAMPLES##################
# from lime import lime_tabular
# import matplotlib.pyplot as plt

# X_train_flat = X_train.reshape(X_train.shape[0], -1)
# X_test_flat = X_test.reshape(X_test.shape[0], -1)

# def predict_fn(input_2d):
#     reshaped = input_2d.reshape(-1, 4, 1000, 1)
#     return model.predict(reshaped)

# feature_names = [f"ch{ch}_t{t}" for ch in range(4) for t in range(1000)]

# explainer = lime_tabular.LimeTabularExplainer(
#     training_data=X_train_flat,
#     feature_names=feature_names,
#     class_names=["Class 0", "Class 1", "Class 2"],
#     discretize_continuous=True,
#     mode='classification'
# )

# num_samples_to_explain = 5
# all_explanations = []

# for sample_index in range(num_samples_to_explain):
#     sample = X_test_flat[sample_index]

#     explanation = explainer.explain_instance(
#         data_row=sample,
#         predict_fn=predict_fn,
#         num_features=10
#     )
    
#     print(f"\n🔍 Explanation for sample {sample_index}:")
#     for feature, weight in explanation.as_list():
#         print(f"{feature}: {weight:.4f}")
    
#     all_explanations.append(explanation)

#     # Optional: plot bar chart for each sample
#     features, weights = zip(*explanation.as_list())
#     colors = ['green' if w > 0 else 'red' for w in weights]
    
#     plt.figure(figsize=(8, 4))
#     plt.barh(features, weights, color=colors)
#     plt.xlabel('Contribution to Prediction')
#     plt.title(f'LIME Feature Contributions - Sample {sample_index}')
#     plt.gca().invert_yaxis()
#     plt.tight_layout()
#     plt.show()


##################GET DEPTHWISE CONVOLUTION LAYERS/WEIGHTS##################
import matplotlib.pyplot as plt
# Identify the depthwise layer (usually index 3)
depthwise_layer = model.get_layer("depthwise_conv2d")  # Adjust the name if necessary
print(f"Depthwise Layer: {depthwise_layer.name}")
weights = depthwise_layer.get_weights()[0]
print(weights)  # shape: (channels, 1, 1, depth_multiplier)

channels = ['Temporal-Parietal (TP-7/TP-8)', 'Temporal-Parietal', 'Frontal (F7/F8)', 'Frontal(F7/F8)']

depth_multiplier = weights.shape[-1]
for i in range(depth_multiplier):
    channel_weights = weights[:, 0, 0, i]
    plt.bar(channels, channel_weights)
    plt.title(f'Separable Conv 2D {i+1}')
    plt.ylabel('Weight')
    plt.ylim(-0.1, 0.1 )
    plt.show()



# #########################PLOT WEIGHTS#########################
# # # Assuming you have a weights array from a SEPARABLE convolution layer
# import matplotlib.pyplot as plt
# import numpy as np

# # Paste your weights here
# weights = np.array([[[[-0.04759058], [0.14078975], [0.15618907], [0.04917106], [-0.09182926], [-0.07955189], [-0.10678853], [-0.04892348], [-0.02687308], [0.09075917], [0.04341062], [-0.08926039], [0.09506204], [0.12610576], [0.01482329], [0.10184837]],
#                      [[-0.05407074], [0.15035357], [0.16608247], [-0.088184], [0.08736247], [0.12209264], [0.11679184], [0.06146294], [0.20836554], [0.09904009], [-0.15688688], [0.03068995], [-0.05040992], [-0.00427406], [0.00722413], [0.07776188]],
#                      [[-0.05540551], [0.0864533], [-0.03760922], [0.06959189], [0.05509878], [0.07758908], [0.01951949], [-0.11898851], [0.18665299], [0.01005757], [-0.0903824], [-0.03638545], [0.11014617], [-0.05837098], [0.03240187], [0.13590375]],
#                      [[-0.01963115], [-0.05878411], [0.1215035], [-0.04224256], [0.13116501], [0.05161517], [0.07743566], [0.08760072], [0.06423445], [-0.15380876], [-0.02449908], [-0.10154338], [0.03785773], [0.06550363], [0.1565032], [-0.02322659]],
#                      [[-0.07661165], [0.0105933], [-0.06749151], [-0.05873454], [0.00476592], [-0.10103828], [0.02587897], [-0.00365399], [0.07885921], [-0.11663838], [-0.1094161], [0.03895539], [0.07899105], [-0.09619886], [0.12613264], [0.15711452]],
#                      [[-0.03639951], [-0.04862214], [0.08757344], [0.06113467], [-0.13762821], [0.06072284], [0.0925509], [-0.04773524], [0.13415569], [-0.07259104], [0.07965323], [0.06136671], [-0.02671669], [0.07647371], [0.16118099], [0.02259038]],
#                      [[0.09998147], [0.00432759], [-0.14163134], [-0.09117699], [-0.09411203], [0.16593891], [0.04341488], [-0.09143326], [0.02285643], [-0.0258196], [-0.10925397], [0.15028264], [0.01339204], [-0.02761232], [-0.05202949], [-0.07419719]],
#                      [[-0.00917723], [-0.00731718], [-0.10983556], [0.02796599], [0.01233764], [0.0173547], [-0.05658041], [0.02020397], [0.02636121], [0.10907193], [-0.02911075], [-0.02798272], [0.10524624], [0.12416995], [0.08076289], [0.04702605]],
#                      [[0.0789308], [-0.08343098], [0.09674664], [-0.08414035], [-0.06830795], [-0.08082672], [-0.06460264], [-0.09376465], [0.04330574], [0.07241144], [-0.05138022], [-0.08089755], [0.07412557], [0.04489509], [-0.16240942], [0.08380292]],
#                      [[0.02483808], [0.0048432], [0.08378723], [0.07131249], [-0.04703815], [-0.06988948], [0.12905866], [0.09059542], [-0.12976031], [-0.1484095], [-0.0366131], [-0.1112763], [0.01397021], [-0.09715839], [-0.05555909], [-0.01777906]],
#                      [[0.01638328], [0.02295901], [0.1434179], [0.09608681], [0.01995214], [-0.02191393], [0.10781877], [0.06192987], [-0.06282343], [-0.04414833], [0.07495473], [0.06315617], [0.03655256], [-0.00063533], [-0.0235457], [-0.06351355]],
#                      [[0.03433074], [0.07865302], [-0.07676893], [-0.00126836], [0.07977214], [-0.04902256], [0.15052304], [0.13574159], [0.03233736], [0.06712709], [0.07679922], [-0.00694715], [-0.06006906], [-0.00364061], [0.0130793], [0.12584005]],
#                      [[-0.14025135], [-0.07953948], [-0.07195245], [-0.12101591], [0.05736764], [-0.03229589], [0.08495231], [-0.05624307], [0.01434445], [0.00369101], [-0.05368471], [0.01861536], [0.01384301], [0.08694826], [0.03541879], [0.00466406]],
#                      [[0.09513744], [0.01427641], [-0.06718739], [-0.13217805], [0.08956868], [-0.10103314], [0.09232226], [0.0960286], [0.07279989], [-0.13340208], [0.00087354], [0.03979361], [0.0177439], [0.10615112], [-0.00353272], [-0.07662558]],
#                      [[-0.04431178], [0.15943977], [0.03415501], [0.03799108], [0.05819354], [0.02685323], [-0.01832939], [0.11106593], [-0.01730475], [0.02679895], [-0.14653605], [0.0858735], [0.069477], [-0.09586761], [-0.04447176], [-0.08168299]],
#                      [[0.01107774], [0.14194725], [0.091565], [0.10388066], [-0.06655801], [-0.01405927], [-0.12528196], [0.11465769], [0.03521409], [0.13362426], [0.1580569], [-0.0985682], [-0.13834067], [0.08019324], [0.01051471], [-0.08941156]]]])

# # Reshape to (16 filters, 16 kernel points)
# filters = weights.squeeze().T

# plt.figure(figsize=(12, 8))
# for i in range(16):
#     plt.plot(filters[i], label=f'Filter {i+1}')
# plt.title('SeparableConv2D Weights (16 Filters, 16 Temporal Points)')
# plt.xlabel('Kernel Index')
# plt.ylabel('Weight')
# plt.grid(True)
# plt.legend(ncol=2, fontsize=9)
# plt.tight_layout()
# plt.show()


#####################Compute averages for various metrics#####################
# Convert one-hot labels back to class indices for metric calculations
# import os
# from sklearn.metrics import confusion_matrix, f1_score, recall_score, accuracy_score
# from sklearn.model_selection import train_test_split

# y_train_labels = np.argmax(y_train, axis=1)
# y_test_labels = np.argmax(y_test, axis=1)

# # Model file list
# model_files = [
#     "eegnet_model_1_averagepooling_50epochs_channelweighted[0011]20250731-2324.h5",
#     "eegnet_model_1_averagepooling_50epochs_channelweighted[1100]20250731-2318.h5",
#     "eegnet_model_1_averagepooling_50epochs_channelweighted[1166]20250731-2243.h5",
#     "eegnet_model_1_averagepooling_50epochs_channelweighted[5501]20250731-2339.h5",
#     "eegnet_model_1_averagepooling_50epochs_channelweighted[6611]20250731-2300.h5",
#     "eegnet_model_1_maxpooling_35epochs20250731-1559.h5",
#     "eegnet_model_1_maxpooling_50epochs20250731-1603.h5",
# ]

# output_dir = "model_random_sample_evaluation_results_unweighted"
# os.makedirs(output_dir, exist_ok=True)

# # === Extract weights from filename like [5501] → [0.5, 0.5, 0, 1.0] ===
# import re
# import numpy as np

# def extract_weights_from_filename(filename):
#     import re
#     match = re.search(r"\[(\d{4})\]", filename)
#     if not match:
#         return np.ones((1, 4, 1, 1))  # fallback: all weights = 1.0

#     digits = match.group(1)
#     weights = [1.0 if d == '1' else int(d)/10.0 for d in digits]
#     return np.array(weights).reshape(1, 4, 1, 1)


# # === Evaluate one model with its correct channel weighting ===
# def evaluate_model(model_path, n_runs=100):
#     print(f"Evaluating {model_path} for {n_runs} runs...")
#     model = load_model(model_path)

#     channel_weights = extract_weights_from_filename(model_path)
#     print(f"Channel weights: {channel_weights.flatten()}")
#     X_train_w = X_train #* channel_weights
#     X_test_w = X_test #* channel_weights

#     train_accs, test_accs = [], []
#     f1s, recalls = [], []
#     cms = []

#     for _ in range(n_runs):
#         idx = np.random.choice(len(X_test_w), int(0.8 * len(X_test_w)), replace=True)
#         X_test_sample = X_test_w[idx]
#         y_test_sample = y_test_labels[idx]
#         print(idx)

#         train_pred_probs = model.predict(X_train_w, verbose=0)
#         train_preds = np.argmax(train_pred_probs, axis=1)
#         train_acc = accuracy_score(y_train_labels, train_preds)
#         train_accs.append(train_acc)

#         test_pred_probs = model.predict(X_test_sample, verbose=0)
#         test_preds = np.argmax(test_pred_probs, axis=1)

#         test_acc = accuracy_score(y_test_sample, test_preds)
#         recall = recall_score(y_test_sample, test_preds, average='macro')
#         f1 = f1_score(y_test_sample, test_preds, average='macro')
#         cm = confusion_matrix(y_test_sample, test_preds, labels=[0, 1, 2])

#         test_accs.append(test_acc)
#         recalls.append(recall)
#         f1s.append(f1)
#         cms.append(cm)
#         print(test_accs)

#     avg_train_acc = np.mean(train_accs)
#     avg_test_acc = np.mean(test_accs)
#     avg_recall = np.mean(recalls)
#     avg_f1 = np.mean(f1s)
#     avg_cm = np.mean(cms, axis=0)

#     return {
#         "train_accuracy": avg_train_acc,
#         "test_accuracy": avg_test_acc,
#         "recall": avg_recall,
#         "f1_score": avg_f1,
#         "avg_confusion_matrix": avg_cm
#     }

# # === Save metrics and confusion matrix ===
# def save_results(results, model_name):
#     base_name = os.path.splitext(os.path.basename(model_name))[0]
#     metrics_file = os.path.join(output_dir, f"{base_name}_metrics.txt")
#     cm_file = os.path.join(output_dir, f"{base_name}_confusion_matrix.npy")

#     with open(metrics_file, "w") as f:
#         for k, v in results.items():
#             if k != "avg_confusion_matrix":
#                 f.write(f"{k}: {v:.4f}\n")

#     np.save(cm_file, results["avg_confusion_matrix"])

#     print(f"Saved metrics to {metrics_file}")
#     print(f"Saved confusion matrix to {cm_file}")

# # === Run for all models ===
# for mf in model_files:
#     results = evaluate_model(mf)
#     save_results(results, mf)