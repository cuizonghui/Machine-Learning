# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from itertools import combinations


def plot_scatter(features, feature_indices, feature_names):
    """
    绘制散点图
    :param features:
    :param feature_indices:
    :param feature_names:
    :return:
    """

    plt.scatter(features[:50, feature_indices[0]], features[:50, feature_indices[1]], label='0')
    plt.scatter(features[50:100, 0], features[50:100, 1], label='1')
    plt.xlabel(feature_names[feature_indices[0]])
    plt.ylabel(feature_names[feature_indices[1]])
    plt.title(str(feature_names[feature_indices[0]]) + ' and ' + str(feature_names[feature_indices[1]]))
    plt.legend()
    plt.show()


def plot_line_scatter(features, feature_indices, feature_names, perceptron):
    """
    绘制分割线和散点图
    :param features:
    :param feature_indices:
    :param feature_names:
    :param perceptron:
    :return:
    """

    x_points = np.linspace(0, 7, 10)
    y_ = -(perceptron.w[0] * x_points + perceptron.b) / perceptron.w[1]
    plt.plot(x_points, y_)

    plt.plot(features[:50, feature_indices[0]], features[:50, feature_indices[1]], 'bo', color='blue', label='0')
    plt.plot(features[50:100, feature_indices[0]], features[50:100, feature_indices[1]], 'bo', color='orange', label='1')
    plt.xlabel(feature_names[feature_indices[0]])
    plt.ylabel(feature_names[feature_indices[1]])
    plt.title(str(feature_names[feature_indices[0]]) + ' and ' + str(feature_names[feature_indices[1]]))
    plt.legend()
    plt.show()


class Perceptron:
    def __init__(self, features, labels):

        self.w = np.zeros(features.shape[1], dtype=np.float32)
        self.b = 0.0
        self.learning_rate = 0.5
        self.features = features
        self.labels = labels

    def fit(self):
        wrong_data_num = 1
        iter_num = 0
        while wrong_data_num:
            wrong_data_num = self.__sgd()
            iter_num += 1
        print('迭代次数', iter_num)
        self.wrong_data_num = wrong_data_num

    def predict(self, feature):
        return self.__sign_fun(feature)



    def __sign_fun(self, feature):
        return np.sign(np.dot(self.w, feature) + self.b)

    def __get_wrong_data(self):
        wrong_data_indices = []
        for index, (feature, label) in enumerate(zip(self.features, self.labels)):
            if (label * self.__sign_fun(feature)) <= 0:
                wrong_data_indices.append(index)
        return wrong_data_indices

    def __sgd(self):
        """
        随机梯度下降
        :return:
        """
        wrong_data_indices = self.__get_wrong_data()
        wrong_data_num = len(wrong_data_indices)
        if wrong_data_num > 0:
            random_data_index = np.random.randint(len(wrong_data_indices))

            self.w = self.w + self.learning_rate * np.dot(self.labels[wrong_data_indices[random_data_index]],
                                                          self.features[wrong_data_indices[random_data_index]])
            self.b = self.b + self.learning_rate * self.labels[wrong_data_indices[random_data_index]]
            return wrong_data_num
        else:
            return wrong_data_num



def main():

    iris = load_iris()

    features = iris.data
    labels = iris.target
    labels = np.array([-1 if i == 0 else 1 for i in labels])

    feature_names = iris.feature_names

    feature_name_combinations = list(combinations([i for i in range(len(feature_names))], 2))
    print(feature_names)

    # for indices in feature_name_combinations:
    #     plot_scatter(features, indices, feature_names)
    import time
    const_time = []
    for indices in feature_name_combinations:
        # plot_scatter(features, indices, feature_names)
        features_select = features[:, indices]
        # features_select=features
        perceptron = Perceptron(features_select, labels)

        start = time.time()
        perceptron.fit()
        end = time.time()
        const_time.append(end - start)
        print('time %5.2f' % (end - start))
        print('w,b', perceptron.w, perceptron.b)
        print('wrong_data_num', perceptron.wrong_data_num)

        # plot_line_scatter(features, feature_indices=indices, feature_names=feature_names, perceptron=perceptron)
    print('mean time', np.mean(const_time))


if __name__ == '__main__':
    main()
