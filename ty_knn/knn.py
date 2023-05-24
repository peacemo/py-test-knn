import numpy as np

# 定义距离度量函数（欧氏距离）
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# KNN算法
def knn(X_train, y_train, X_test, K):
    predictions = []
    for i in range(X_test.shape[0]):
        x_test = X_test[i, :]
        # 计算距离
        distances = [euclidean_distance(x_test, x_train) for x_train in X_train]
        # 选择最近的K个样本
        k_nearest = np.argsort(distances)[:K]
        # 标签投票
        votes = {}
        for idx in k_nearest:
            label = y_train[idx]
            votes[label] = votes.get(label, 0) + 1
        # 找出票数最多的标签
        predicted_label = max(votes, key=votes.get)
        predictions.append(predicted_label)
    return predictions
