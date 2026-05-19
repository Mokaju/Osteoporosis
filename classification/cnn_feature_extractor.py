""" Shared utils for CNN feature extraction from saved models. """

import os
import numpy as np
from tensorflow.keras.models import load_model, Sequential


def build_feature_extractor(model_path, drop_last_n=4):
    base_model = load_model(model_path)
    feature_extractor = Sequential()
    for layer in base_model.layers[:-drop_last_n]:
        feature_extractor.add(layer)
    return feature_extractor


def extract_features(feature_extractor, x):
    return feature_extractor.predict(x)


def load_split_arrays(npz_path):
    data = np.load(npz_path)
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']


def get_cnn_model_paths(model_dir):
    return [
        os.path.join(model_dir, f)
        for f in os.listdir(model_dir)
        if f.endswith('.h5')
    ]