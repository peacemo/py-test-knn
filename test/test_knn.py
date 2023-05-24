import numpy as np
from ty_knn import knn
import pytest

def test_knn():
    # 示例用法
    X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    y_train = np.array(['A', 'B', 'A', 'B', 'A'])
    X_test = np.array([[4, 5], [2, 3], [8, 9]])
    K = 3

    predicted_labels = knn(X_train, y_train, X_test, K)

    assert isinstance(predicted_labels, list)
