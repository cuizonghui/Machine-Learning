# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------

import numpy as np
from sklearn.datasets import load_iris
from collections import Counter
import os


class NaiveBayesClassification:
    """朴素贝叶斯分类"""

    def fit(self, features, labels, feature_names):
        if len(features) == 0 or len(features.shape) != 2:
            raise ValueError('features必须是samplesXfeatures的形式！')
        self.features = features
        self.labels = labels
        self.class_frequency_dict = dict(Counter(labels))
        self.feature_names = feature_names
        self.prior_probability_dict = self.prior_probability()
        self.posterior_probability_dict = self.posterior_probability()

    def predict(self, features):
        if len(features) == 0:
            raise ValueError('features必须是非空的形式！')
        if len(features.shape) == 1:
            self.features = np.array(features).reshape(1, len(features))
        elif len(features.shape) == 2:
            self.features = features
        else:
            raise ValueError('features必须是samplesXfeatures的形式！')
        predict_classes = []
        for feature in self.features:
            feature = np.reshape(feature, [1, len(feature)])
            probility_list = []
            probility_dict = {}
            for class_value in self.class_dict.keys():
                for index in range(feature.shape[1]):
                    select_feature_name = self.feature_names[index] + '=' + str(feature[0, index]) + "|" + class_value
                    if select_feature_name in self.posterior_probability_dict.keys():
                        select_feauture_posterior_probability = self.posterior_probability_dict[select_feature_name]
                        # 下一行有错误，classvalue不对
                        probility = self.prior_probability_dict[class_value] * select_feauture_posterior_probability
                        probility_list.append(probility)
                        probility_dict[class_value] = probility

            predict_result = max(probility_dict.items(), key=lambda x: x[1])[0]
            predict_class = int(predict_result.split('=')[1])
            predict_classes.append(predict_class)
        return predict_classes

    def prior_probability(self):
        return {'class=' + str(key): value / sum_frequency for key, value in self.class_frequency_dict.items()
                for sum_frequency in [np.sum(list(self.class_frequency_dict.values()))]}

    def posterior_probability(self):

        self.class_dict = {'class=' + str(key): np.where(self.labels == key)[0]
                           for key in self.class_frequency_dict.keys()}
        posterior_probability_dict = {}
        for feature_index in range(self.features.shape[1]):
            feature_dict = Counter(self.features[:, feature_index])
            feature_dict = {self.feature_names[feature_index] + '=' + str(key):
                            np.where(self.features[:, feature_index] == key)[0]
                            for key in feature_dict.keys()}

            for class_key, class_indices_value in self.class_dict.items():
                for feature_key, feature_value in feature_dict.items():
                    intersect_indices = np.intersect1d(class_indices_value, feature_value)
                    posterior_probability = np.true_divide(len(intersect_indices), len(class_indices_value))
                    posterior_probability_dict[feature_key + '|' + class_key] = posterior_probability

        # 这是用推导式代替上边循环的实现
        # feature_dicts = [
        #     {feature_name[feature_index] + '=' + str(key): np.where(features[:, feature_index] == key)[0]
        #      for key in Counter(features[:, feature_index]).keys()}
        #     for feature_index in range(features.shape[1])]
        # posterior_probability_dict1 = {class_key + ',' + feature_key:
        #                                np.true_divide(len(np.intersect1d(class_value, feature_value)), len(class_value))
        #                                for feature_dict in feature_dicts
        #                                for class_key, class_value in class_dict.items()
        #                                for feature_key, feature_value in feature_dict.items()}

        return posterior_probability_dict


class BayesClassification:
    """贝叶斯分类"""

    def fit(self, features, labels, feature_names):
        if len(features) == 0 or len(features.shape) != 2:
            raise ValueError('features必须是samplesXfeatures的形式！')
        self.features = features
        self.labels = labels
        self.class_frequency_dict = dict(Counter(labels))
        self.feature_names = feature_names
        self.prior_probability_dict = self.prior_probability()
        self.posterior_probability_dict = self.posterior_probability()

    def predict(self, features):
        if len(features) == 0:
            raise ValueError('features必须是非空的形式！')
        if len(features.shape) == 1:
            self.features = np.array(features).reshape(1, len(features))
        elif len(features.shape) == 2:
            self.features = features
        else:
            raise ValueError('features必须是samplesXfeatures的形式！')
        predict_classes = []
        for feature in self.features:
            feature = np.reshape(feature, [1, len(feature)])
            probility_list = []
            probility_dict = {}
            for class_value in self.class_dict.keys():
                for index in range(feature.shape[1]):
                    select_feature_name = self.feature_names[index] + '=' + str(feature[0, index]) + "|" + class_value
                    if select_feature_name in self.posterior_probability_dict.keys():
                        select_feauture_posterior_probability = self.posterior_probability_dict[select_feature_name]
                        # 下一行有错误，classvalue不对
                        probility = self.prior_probability_dict[class_value] * select_feauture_posterior_probability
                        probility_list.append(probility)
                        probility_dict[class_value] = probility

            predict_result = max(probility_dict.items(), key=lambda x: x[1])[0]
            predict_class = int(predict_result.split('=')[1])
            predict_classes.append(predict_class)
        return predict_classes

    def prior_probability(self):
        return {'class=' + str(key): value / sum_frequency for key, value in self.class_frequency_dict.items()
                for sum_frequency in [np.sum(list(self.class_frequency_dict.values()))]}

    def posterior_probability(self):

        self.class_dict = {'class=' + str(key): np.where(self.labels == key)[0]
                           for key in self.class_frequency_dict.keys()}
        posterior_probability_dict = {}
        for feature_index in range(self.features.shape[1]):
            feature_dict = Counter(self.features[:, feature_index])
            feature_dict = {self.feature_names[feature_index] + '=' + str(key):
                            np.where(self.features[:, feature_index] == key)[0]
                            for key in feature_dict.keys()}

            for class_key, class_indices_value in self.class_dict.items():
                for feature_key, feature_value in feature_dict.items():
                    intersect_indices = np.intersect1d(class_indices_value, feature_value)
                    posterior_probability = np.true_divide(
                        len(intersect_indices) + 1, len(class_indices_value) + len(self.class_frequency_dict))
                    posterior_probability_dict[feature_key + '|' + class_key] = posterior_probability

        # 这是用推导式代替上边循环的实现
        # feature_dicts = [
        #     {feature_name[feature_index] + '=' + str(key): np.where(features[:, feature_index] == key)[0]
        #      for key in Counter(features[:, feature_index]).keys()}
        #     for feature_index in range(features.shape[1])]
        # posterior_probability_dict1 = {class_key + ',' + feature_key:
        #                                np.true_divide(len(np.intersect1d(class_value, feature_value)), len(class_value))
        #                                for feature_dict in feature_dicts
        #                                for class_key, class_value in class_dict.items()
        #                                for feature_key, feature_value in feature_dict.items()}

        return posterior_probability_dict


def main():
    iris = load_iris()
    features = iris['data']
    labels = iris['target']
    feature_names = [feature_name[:-5] for feature_name in iris['feature_names']]
    print(np.unique(labels))

    naive_bayes_classification = NaiveBayesClassification()
    naive_bayes_classification.fit(features, labels, feature_names)
    predict_class = naive_bayes_classification.predict(features)
    indices = [index for index in range(len(predict_class)) if predict_class[index] != labels[index]]
    print("多项分布朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (iris.data.shape[0], np.sum((iris.target != predict_class))))
    print(indices)

    bayes_classification = BayesClassification()
    bayes_classification.fit(features, labels, feature_names)
    predict_class = bayes_classification.predict(features)
    print("多项分布贝叶斯，样本总数： %d 错误样本数 : %d" % (iris.data.shape[0], np.sum((iris.target != predict_class))))
    indices = [index for index in range(len(predict_class)) if predict_class[index] != labels[index]]
    print(indices)

    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf = clf.fit(features, labels)
    y_pred = clf.predict(features)
    print("sklearn多项分布朴素贝叶斯，样本总数： %d 错误样本数 : %d" % (iris.data.shape[0], np.sum((iris.target != y_pred))))
    indices = [index for index in range(len(y_pred)) if y_pred[index] != labels[index]]
    print(indices)


if __name__ == '__main__':
    main()
