# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:02:14 2025

@author: omer
"""

import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from mlxtend.plotting import plot_confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from datetime import datetime
import random

# --------------------
# DATA LOADING
# --------------------
def load_images_from_folder(folder, label, size=(224, 224)):
    images, labels = [], []
    for filename in os.listdir(folder):
        try:
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.resize(img, size)
            images.append(img)
            labels.append(label)
        except:
            continue
    return np.array(images), labels

# --------------------
# GRAD-CAM FUNCTION
# --------------------
def generate_gradcam(model, img_array, class_index, layer_name='conv5_block3_out'):
    grad_model = Model([model.inputs], [model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img_array, axis=0))
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)[0]
    conv_outputs = conv_outputs[0]

    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * conv_outputs[:, :, i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))  # düzeltildi
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    return cam

def overlay_gradcam(img, cam):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.uint8(img * 255), 0.5, heatmap, 0.5, 0)
    return overlay

# --------------------
# PATHS
# --------------------
path_0 = r"E:\Muhammed_hoca_polip\calisma\Goruntuler\PARANAZAL"
path_1 = r"E:\Muhammed_hoca_polip\calisma\Goruntuler\Mukormukozis"
path_2 = r"E:\Muhammed_hoca_polip\calisma\Goruntuler\Polip"

# Load data
X0, y0 = load_images_from_folder(path_0, 0)
X1, y1 = load_images_from_folder(path_1, 1)
X2, y2 = load_images_from_folder(path_2, 2)

X = np.concatenate((X0, X1, X2), axis=0)
y = np.array(y0 + y1 + y2)
X = X.astype("float32") / 255.0

# --------------------
# K-FOLD
# --------------------
k = 5
skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

acc_scores, prec_scores, rec_scores, f1_scores = [], [], [], []
fold = 1

base_result_dir = "results"
os.makedirs(base_result_dir, exist_ok=True)

for train_val_idx, test_idx in skf.split(X, y):
    print(f"\n--- Fold {fold} ---")
    fold_dir = os.path.join(base_result_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Split
    X_trainval, X_test = X[train_val_idx], X[test_idx]
    y_trainval, y_test = y[train_val_idx], y[test_idx]

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.2, stratify=y_trainval, random_state=42
    )

    y_train_cat = to_categorical(y_train, num_classes=3)
    y_val_cat = to_categorical(y_val, num_classes=3)
    y_test_cat = to_categorical(y_test, num_classes=3)

    # Model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False

    x = Flatten()(base_model.output)
    x = Dense(1024, activation="relu", kernel_regularizer=l2(1e-5))(x)
    output = Dense(3, activation="softmax", kernel_regularizer=l2(1e-5))(x)
    model = Model(inputs=base_model.input, outputs=output)

    optimizer = Adam(learning_rate=1e-4)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )

    # Callbacks
    checkpoint_path = os.path.join(fold_dir, "best_model.h5")
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
    log_dir = os.path.join(fold_dir, "logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard = TensorBoard(log_dir=log_dir)

    # Train
    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=50,
        batch_size=16,
        callbacks=[early_stop, checkpoint, tensorboard],
        verbose=1
    )

    np.save(os.path.join(fold_dir, "training_history.npy"), history.history)

    # Evaluation
    model.load_weights(checkpoint_path)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    acc_scores.append(report['accuracy'])
    prec_scores.append(report['macro avg']['precision'])
    rec_scores.append(report['macro avg']['recall'])
    f1_scores.append(report['macro avg']['f1-score'])

    with open(os.path.join(fold_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_true, y_pred, target_names=['Normal', 'Mucormycosis', 'Polyp']))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plot_confusion_matrix(conf_mat=cm)
    ax.set_xticklabels([''] + ['Normal', 'Mucormycosis', 'Polyp'], rotation=45)
    ax.set_yticklabels([''] + ['Normal', 'Mucormycosis', 'Polyp'])
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"))
    plt.close()

    # Grad-CAM (1 örnek)
    sample_idx = random.choice(range(len(X_test)))
    sample_img = X_test[sample_idx]
    sample_true = y_true[sample_idx]
    sample_pred = y_pred[sample_idx]

    cam = generate_gradcam(model, sample_img, class_index=sample_pred)
    gradcam_overlay = overlay_gradcam(sample_img, cam)
    output_path = os.path.join(fold_dir, f"gradcam_{sample_idx}.png")
    cv2.imwrite(output_path, gradcam_overlay)

    fold += 1

# --------------------
# FINAL SUMMARY
# --------------------
print("\n--- Final 5-Fold Average Results ---")
print(f"Accuracy: {np.mean(acc_scores) * 100:.2f}% ± {np.std(acc_scores) * 100:.2f}")
print(f"Precision: {np.mean(prec_scores):.4f}")
print(f"Recall: {np.mean(rec_scores):.4f}")
print(f"F1 Score: {np.mean(f1_scores):.4f}")