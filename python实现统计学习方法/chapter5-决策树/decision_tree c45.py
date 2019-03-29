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
from collections import Counter


def create_data():

    features = [['青年', '否', '否', '一般'],
                ['青年', '否', '否', '好'],
                ['青年', '是', '否', '好'],
                ['青年', '是', '是', '一般'],
                ['青年', '否', '否', '一般'],
                ['中年', '否', '否', '一般'],
                ['中年', '否', '否', '好'],
                ['中年', '是', '是', '好'],
                ['中年', '否', '是', '非常好'],
                ['中年', '否', '是', '非常好'],
                ['老年', '否', '是', '非常好'],
                ['老年', '否', '是', '好'],
                ['老年', '是', '否', '好'],
                ['老年', '是', '否', '非常好'],
                ['老年', '否', '否', '一般'],
                ]
    labels = [u'否', u'否', u'是', u'是', u'否', '否', '否', '是', '是', '是', '是', '是', '是', '是', '否']
    feature_name = ['年龄', '有工作', '有自己的房子', '信贷情况', u'类别']
    dataset_dict = {'features': features,
                    'labels': labels,
                    'feature_name': feature_name}

    # 返回数据集和每个维度的名称
    return dataset_dict


class Node:
    def __init__(self, root=True, label=None, feature_name=None, feature=None):
        self._root = root
        self._label = label
        self._feature_name = feature_name
        self._feature = feature
        self._tree = {}
        self.result = {'tree': self._tree,
                       'feature': self._feature,
                       'label': self._label,
                       'feature_name': self._feature_name}

    def __repr__(self):
        return '{}'.format(self.result)

    def add_node(self, val, node):
        self._tree[val] = node

    def predict(self, features):
        if self._root is True:
            return self._label
        return self._tree[features[self._feature].predict(features)]


class DecisionTreeClassificationC45:
    def __init__(self, epsilon=0.01):
        self._epsilon = epsilon
        self._tree = {}

    def fit(self, features, labels, features_name):

        # 如果features为空，则Ｔ为单节点树，将label中最大的类别Ｃｋ设置为节点类别，返回Ｔ
        if len(features) == 0:
            labels_frequency_dict = dict(Counter(labels))
            return Node(root=True, label=max(labels_frequency_dict, key=labels_frequency_dict.get))

        if not isinstance(features, np.ndarray):
            features = np.array(features)
            labels = np.array(labels)
            features_name = np.array(features_name)

        # 如果所有实例同属于一类Ｃｋ,则Ｔ为单节点树，并将类节点Cｋ作为节点的类标记，返回Ｔ
        if len(np.unique(labels)) == 1:
            return Node(root=True, label=labels[0])

        best_feature_name, best_feature_index, best_feature_info_gain_ratio = self._get_best_feature(
            features, labels, features_name)

        # 信息增益小于阈值epsilon则设置为单节点树，并讲实例中最大类别Ｃｋ设置为节点类别，返回Ｔ
        if best_feature_info_gain_ratio < self._epsilon:
            labels_frequency_dict = dict(Counter(labels))
            return Node(root=True, label=max(labels_frequency_dict, key=labels_frequency_dict.get))

        # 构建Ag子集
        node_tree = Node(root=False, feature_name=best_feature_name, feature=best_feature_index)
        feature_list = np.unique(features[:, best_feature_index])
        for f in feature_list:
            indices = np.where(features[:, best_feature_index] == f)[0]
            # features_copy=features.copy()
            features_copy = np.delete(features, best_feature_index, axis=1)
            sub_features = features_copy[indices]
            sub_labels = labels[indices]
            sub_features_name = np.delete(features_name, best_feature_index, axis=0)
            sub_tree = self.fit(sub_features, sub_labels, sub_features_name)
            node_tree.add_node(f, sub_tree)
        return node_tree

    def _get_best_feature(self, features, labels, features_name):
        information_entropy = self._calculate_information_entropy(labels)
        information_gain_ratio_dict = {}
        for index in range(len(features[0])):
            feature = features[:, index]
            condition_entropy_value = self._calculate_condition_entropy(feature, labels)
            information_gain_ratio_dict[features_name[index]] = self._information_gain_ratio(information_entropy,
                                                                                 condition_entropy_value)
        best_feature_name = max(information_gain_ratio_dict, key=information_gain_ratio_dict.get)
        best_feature_index = np.where(features_name == best_feature_name)[0][0]
        return best_feature_name, best_feature_index, information_gain_ratio_dict[best_feature_name]

    def _information_gain_ratio(self, information_entropy, condition_entropy):
        """
        信息增益
        :param information_entropy: 信息熵
        :param condition_entropy: 条件信息熵
        :return:
        """
        return (information_entropy - condition_entropy)/information_entropy

    def _calculate_information_entropy(self, labels):
        """
        计算信息熵
        :param labels:
        :return:
        """
        class_frequency_dict = dict(Counter(labels))
        class_example_num = len(labels)
        entropy_value = np.sum(list(map(
            lambda x: -((x / class_example_num) * np.log2(x / class_example_num)),
            list(class_frequency_dict.values()))))

        return entropy_value

    def _calculate_condition_entropy(self, feature, labels):
        """
        计算条件信息熵
        :param feature: 特征
        :param labels: ｌａｂｅｌｓ
        :return:
        """
        feature_frequency_dict = Counter(feature)
        probability_dict = self._frequency_dict_to_probability_dict(feature_frequency_dict)
        feature_labels_dict = {key: labels[np.where(feature == key)[0]] for key in feature_frequency_dict.keys()}
        # 这一行应该分开写
        # condition_entropy_value_dict=self._calculate_information_entropy(feature_labels_dict)
        condition_entropy_value = 0
        for key in feature_labels_dict.keys():
            condition_entropy_value += probability_dict[key] * self._calculate_information_entropy(feature_labels_dict[key])

        return condition_entropy_value

    def _frequency_dict_to_probability_dict(self, frequency_dict):
        return {key: value / (np.sum(list(frequency_dict.values()))) for key, value in frequency_dict.items()}










def main():
    dataset = create_data()
    features = dataset['features']
    labels = dataset['labels']
    features_name = dataset['feature_name']

    decision_tree_classification = DecisionTreeClassificationC45()
    tree = decision_tree_classification.fit(features, labels, features_name)
    print(tree)


if __name__ == '__main__':
    main()
