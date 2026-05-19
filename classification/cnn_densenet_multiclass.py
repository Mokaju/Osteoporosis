#!/usr/bin/env python
# coding: utf-8
"""Train 3‑class CNN models (CNN‑DenseNet named) with 5‑fold cross‑val."""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


def load_split_data(npz_path='train_test_split_arrays_3_classes.npz'):
    data = np.load(npz_path)
    x_train = data['X_train']
    x_test = data['X_test']
    y_train = data['Y_train']
    y_test = data['Y_test']
    return x_train, x_test, y_train, y_test


def create_model(input_shape=(100, 200, 1), n_classes=3):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(
            32, (3, 3), activation='relu',
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )
    return model


def compute_weighted_multiclass_metrics(cm):
    support = cm.sum(axis=1).astype(float)
    tp = np.diag(cm).astype(float)  # TP per class
    fp = np.sum(cm, axis=0).astype(float) - tp
    fn = np.sum(cm, axis=1).astype(float) - tp
    tn = cm.sum() - (fp + fn + tp)

    # Per‑class metrics with safe division
    precision = np.divide(
        tp, tp + fp,
        out=np.zeros_like(tp),
        where=(tp + fp) != 0
    )
    recall = np.divide(
        tp, tp + fn,
        out=np.zeros_like(tp),
        where=(tp + fn) != 0
    )  # recall = sensitivity
    specificity = np.divide(
        tn, tn + fp,
        out=np.zeros_like(tp),
        where=(tn + fp) != 0
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0
    )
    accuracy_per_class = np.divide(
        tp + tn, tp + fp + fn + tn,
        out=np.zeros_like(tp),
        where=(tp + fp + fn + tn) != 0
    )

    total_support = support.sum()
    weighted_precision = np.sum(precision * support) / total_support
    weighted_recall = np.sum(recall * support) / total_support
    weighted_specificity = np.sum(specificity * support) / total_support
    weighted_f1 = np.sum(f1 * support) / total_support
    weighted_accuracy = np.sum(accuracy_per_class * support) / total_support

    return {
        "accuracy": weighted_accuracy,
        "sensitivity": weighted_recall,
        "specificity": weighted_specificity,
        "precision": weighted_precision,
        "f1_score": weighted_f1,
    }


def main(
    npz_path='train_test_split_arrays_3_classes.npz',
    save_dir='3-class-classification models/CNN-Densenet',
    epochs=100,
    batch_size=32
):
    os.makedirs(save_dir, exist_ok=True)

    print("Loading 3‑class train/test split...")
    x_train, x_test, y_train, y_test = load_split_data(npz_path)

    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(x_train, y_train)):
        x_tr = x_train[tr_idx]
        y_tr = y_train[tr_idx]
        x_vl = x_train[val_idx]
        y_vl = y_train[val_idx]

        print(f"Fold {fold}: building and training 3‑class CNN...")
        model = create_model()

        history = model.fit(
            x_tr, y_tr,
            batch_size=batch_size,
            validation_data=(x_vl, y_vl),
            verbose=1,
            epochs=epochs
        )

        # Save history
        pd.DataFrame(history.history).to_csv(
            os.path.join(save_dir, f'CNN-DenseNet_training_history_fold_{fold}_3_classes.csv'),
            index=False
        )

        # Save model
        model_path = os.path.join(
            save_dir,
            f'CNN-DenseNet_model_fold_{fold}_for_3_classes.h5'
        )
        model.save(model_path)

        # Test‑set evaluation
        y_pred = model.predict(x_test)  # (N, 3) probabilities
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_test_classes, y_pred_classes)
        pd.DataFrame(cm).to_csv(
            os.path.join(save_dir, f'CNN-DenseNet_confusion_matrix_fold_{fold}_3_classes.csv'),
            index=False
        )

        metrics = compute_weighted_multiclass_metrics(cm)
        pd.DataFrame([metrics]).to_csv(
            os.path.join(save_dir, f'CNN-DenseNet_metrics_fold_{fold}_3_classes.csv'),
            index=False
        )


if __name__ == '__main__':
    main()