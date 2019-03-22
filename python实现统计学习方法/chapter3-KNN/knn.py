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
from collections import Counter
np.seterr(invalid='ignore')

class KNN:
    # def __init__(self, features, labels):
    #     self.features = features
    #     self.labels = labels

    def fit(self, features, labels):
        self.features = features
        self.labels = labels


    def predict(self, test_feature, p, k):
        lp_dis_list = []
        for feature in self.features:
            lp_dis = self.__lp_distance(test_feature, feature, p)
            lp_dis_list.append(lp_dis)
        sort_indices = np.argsort(lp_dis_list)
        k_nearest_neighbor_indices = sort_indices[:k]
        k_nearest_neighbor_labels = self.labels[k_nearest_neighbor_indices]
        labels_dict = Counter(k_nearest_neighbor_labels)
        predict_class = sorted(labels_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
        return predict_class

    def __lp_distance(self, x0, x1, p):
        sum_value = np.sum(np.power(np.subtract(np.array(x0), np.array(x1)), p))
        return np.power(sum_value, 1.0 / (1.0 * p))


def main():
    iris = load_iris()
    features = iris['data']
    labels = iris['target']

    test = np.random.randint(10, size=4)
    # test = test.reshape(1, len(test))
    knn_classification = KNN()
    knn_classification.fit(features, labels)
    predict_class = knn_classification.predict(test, 3, 10)
    print('test　feature',test,'predict_class',predict_class)

if __name__=="__main__":
    main()


