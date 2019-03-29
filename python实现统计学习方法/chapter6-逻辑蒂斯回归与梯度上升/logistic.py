# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split


def load_data():
    iris = load_iris()
    features = iris['data']
    labels = iris['target']
    labels[labels > 1] = 1
    feature_names = [feature_name[:-5] for feature_name in iris['feature_names']]
    return features, labels, feature_names


class LogisticClassification:
    def __init__(self, learning_rate=0.001, epoch=100):
        self._learning_rate = learning_rate
        self._epoch = epoch

    def fit(self, features, labels):
        self._features = np.array(features)
        self._sample_num, self._feature_num = self._features.shape
        self._labels = np.array(labels)
        self._weight = (np.ones(self._feature_num, dtype=np.float32)).transpose()
        for index in range(self._epoch):
            self.gradient_ascent(self._features, self._labels)

    def predict(self, features, threshold=0.5):
        if len(features) == 1:
            features = np.array(features).reshape([1, len(features)])
        else:
            features = np.array(features)
        if np.array(features).shape[1] != self._feature_num:
            raise ValueError('输入样本特征数与模型所需特征(%d个)不符！' % self._feature_num)

        predicts = np.array([1 if self._sigmoid(np.dot(self._weight, feature)) >= threshold else 0 for feature in features])
        return predicts

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient_ascent(self, features, labels):
        h = self._sigmoid(np.matmul(features, self._weight))
        error = labels - h
        self._weight += self._learning_rate * np.matmul(error, features)


def main():
    features, labels, feature_names = load_data()

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.7)

    logistic_classification = LogisticClassification()
    logistic_classification.fit(train_features, train_labels)
    predicts = logistic_classification.predict(test_features)
    print('logistic accuracy', np.mean(predicts == test_labels))

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=100)
    clf.fit(train_features, train_labels)
    predicts = clf.predict(test_features)
    print('sklearn_logistic accuracy', np.mean(predicts == test_labels))


if __name__ == '__main__':
    main()
