""" Run the full CNN + SVM + RF pipeline. """

from cnn_densenet_binary import main as train_cnn
from cnn_svc import main as train_svc
from cnn_rf import main as train_rf


if __name__ == '__main__':
    print("=== Training CNN‑DenseNet binary models ===")
    train_cnn()

    print("=== Training CNN‑SVM models ===")
    train_svc()

    print("=== Training CNN‑RF models ===")
    train_rf()