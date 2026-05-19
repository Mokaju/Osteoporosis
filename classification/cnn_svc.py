""" Train SVM on CNN‑extracted features. """

import os
import joblib
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from cnn_feature_extractor import (
    build_feature_extractor,
    extract_features,
    load_split_arrays,
    get_cnn_model_paths
)


def compute_binary_metrics(y_true, y_pred):
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
    model_dir='Binary-classification models/CNN-Densenet',
    split_npz='Binary-classification models/CNN-Densenet/train_test_split_arrays.npz',
    save_dir='Binary-classification models/CNN-SVM'
):
    os.makedirs(save_dir, exist_ok=True)

    X_train, X_test, y_train, y_test = load_split_arrays(split_npz)
    cnn_paths = get_cnn_model_paths(model_dir)

    for m, path in enumerate(cnn_paths):
        feature_extractor = build_feature_extractor(path)

        cnn_x_train = extract_features(feature_extractor, X_train)
        cnn_x_test = extract_features(feature_extractor, X_test)

        svm_classifier = SVC(kernel='rbf', C=1, gamma=0.001, probability=True)
        svm_classifier.fit(cnn_x_train, y_train)

        y_pred = svm_classifier.predict(cnn_x_test)

        joblib.dump(
            svm_classifier,
            os.path.join(save_dir, f'svm_classifier_model_for_fold{m}.pkl')
        )

        cm, metrics = compute_binary_metrics(y_test, y_pred)
        pd.DataFrame(cm).to_csv(
            os.path.join(save_dir, f'svm_classifier_confusion_matrix_for_fold{m}.csv'),
            index=False
        )
        pd.DataFrame([metrics]).to_csv(
            os.path.join(save_dir, f'CNN-SVM_metrics_fold_{m}_for_binary_classes.csv'),
            index=False
        )


if __name__ == '__main__':
    main()