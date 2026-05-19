#!/usr/bin/env python
# coding: utf-8
"""Binary CNN training script for DenseNet‑style binary classification."""

import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


def load_and_preprocess_data(datadir='data/', csv_path='filenames_labels.csv'):
    df = pd.read_csv(csv_path, header=None)
    data = []
    labels = []
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    for i in range(len(df)):
        if df.iloc[i, 1] != 's':  # Exclude 's' (shadow images)
            im = cv2.imread(os.path.join(datadir, df.iloc[i, 0]), 0)
            im = cv2.resize(im, (400, 200))  # 400×200
            im = clahe.apply(im)
            im = np.expand_dims(im, axis=-1)  # (H, W, 1)
            data.append(im)
            labels.append(int(df.iloc[i, 1]))

    return np.array(data), np.array(labels)


def augment_and_crop_and_normalize(data, labels):
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        fill_mode='constant',
        cval=0
    )
    aug_data = []
    aug_labels = []
    n = 0
    for x, y in datagen.flow(data, labels, batch_size=10, shuffle=False):
        aug_data.append(x)
        aug_labels.append(y)
        n += 1
        if n == data.shape[0]:
            break
    aug_data = np.concatenate(aug_data)
    aug_labels = np.concatenate(aug_labels)

    final_data = []
    final_labels = []
    for i in range(aug_data.shape[0]):
        x = aug_data[i][50:150, 100:300]  # 100×200 region
        x = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)  # Min‑max
        final_data.append(x)
        y = aug_labels[i]
        if y == 1:
            final_labels.append(0)  # 1 → 0
        else:
            final_labels.append(1)  # 2 / others → 1
    return np.array(final_data), np.array(final_labels)


def create_model(input_shape=(100, 200, 1)):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.25),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss=keras.losses.binary_crossentropy,
        metrics=['accuracy']
    )
    return model


def compute_binary_metrics(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0.0
    f1_score = (2 * precision * sensitivity) / (precision + sensitivity) \
        if (precision + sensitivity) != 0 else 0.0
    return cm, {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1_score": f1_score
    }


def main(
    datadir='data/',
    csv_path='filenames_labels.csv',
    output_dir='Binary-classification models/CNN-Densenet',
    epochs=100,
    batch_size=32
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data...")
    data, labels = load_and_preprocess_data(datadir, csv_path)
    print("Data shape before augmenting:", data.shape, labels.shape)

    print("Augmenting and normalizing...")
    final_data, final_labels = augment_and_crop_and_normalize(data, labels)
    print("Data shape after augmenting:", final_data.shape, final_labels.shape)

    from sklearn.model_selection import train_test_split, KFold
    X = final_data
    y = np.array(final_labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=1
    )
    np.savez(
        os.path.join(output_dir, 'train_test_split_arrays.npz'),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )

    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    m = 0
    for tr_idx, val_idx in kf.split(X_train, y_train):
        x_train = X_train[tr_idx]
        y_train_fold = y_train[tr_idx]
        x_val = X_train[val_idx]
        y_val = y_train[val_idx]

        model = create_model()
        print(f"Fold {m}: training model...")
        history = model.fit(
            x_train,
            y_train_fold,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=1,
            epochs=epochs
        )

        # Save history
        pd.DataFrame(history.history).to_csv(
            os.path.join(output_dir, f'CNN-DenseNet_training_history_fold_{m}.csv'),
            index=False
        )

        # Save model
        model_path = os.path.join(
            output_dir,
            f'CNN-DenseNet_model_fold_{m}_for_binary_classes.h5'
        )
        model.save(model_path)

        # Test set evaluation
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba[:, 0] >= 0.5).astype(int)

        cm, metrics = compute_binary_metrics(y_test, y_pred)

        np.save(
            os.path.join(output_dir, f'CNN-DenseNet_confusion_matrix_fold_{m}_for_binary_classes.npy'),
            cm
        )
        pd.DataFrame(cm).to_csv(
            os.path.join(output_dir, f'CNN-DenseNet_confusion_matrix_fold_{m}_for_binary_classes.csv'),
            index=False
        )
        pd.DataFrame([metrics]).to_csv(
            os.path.join(output_dir, f'CNN-DenseNet_metrics_fold_{m}_for_binary_classes.csv'),
            index=False
        )

        m += 1


if __name__ == '__main__':
    main()