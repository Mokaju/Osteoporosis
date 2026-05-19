#!/usr/bin/env python
# coding: utf-8
"""Run the full CNN‑DenseNet + SVM + RF pipeline for 3 classes."""

from cnn_densenet_multiclass import main as train_cnn_multiclass
from cnn_svc_multiclass import main as train_svc_multiclass
from cnn_rf_multiclass import main as train_rf_multiclass


if __name__ == '__main__':
    print("=== Training 3‑class CNN‑DenseNet models ===")
    train_cnn_multiclass()

    print("=== Training CNN‑SVM models (3‑class) ===")
    train_svc_multiclass()

    print("=== Training CNN‑RF models (3‑class) ===")
    train_rf_multiclass()