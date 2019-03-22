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
iris = load_iris()
features = iris['data']
labels = iris['target']


class Node:
    def __init__(self, item=None, label=None, dim=None, parent=None, left_child=None, right_child=None):
        self.item = item  # 样本信息
        self.item = item  # 结点的值(样本信息)
        self.label = label  # 结点的标签
        self.dim = dim  # 结点的切分的维度(特征)
        self.parent = parent  # 父结点
        self.left_child = left_child  # 左子树
        self.right_child = right_child  # 右子树


class KdTree:
    def __init__(self):
        self.KdTree = None
        self.n = 0
        self.nearest = None
    def create(self, features, labels, depth=0, root_dim=-1):
        if len(features) > 0:
            root_dim = self.__get_root_dim(features, root_dim)
            print(root_dim)
            features_argsort, labels_argsort = self.__get_argsort_features_labels(features, labels, root_dim)
            mid_index = int(len(features) / 2)
            node = Node(features_argsort[mid_index], labels_argsort[mid_index],dim=root_dim)
            if depth == 0:
                self.node = node
            node.left_child = self.create(features_argsort[:mid_index], labels_argsort[:mid_index], depth + 1, root_dim)
            node.right_child = self.create(features_argsort[mid_index + 1:], labels_argsort[mid_index + 1:], depth + 1, root_dim)
            return node
        return None
    def preOrder(self, node):
        if node is not None:
            print(node.depth, node.data)
            self.preOrder(node.left_child)
            self.preOrder(node.right_child)
    def __get_root_dim(self, features, root_dim):
        """
        选取方差最大的特征进行分割，如果该特征的索引与父节点索引相等就选取方差第二大的特征分割
        :param features:
        :param root_dim:
        :return:
        """
        std_value = np.std(features, axis=0)
        std_value_sort = np.argsort(std_value)
        if std_value_sort[-1] != root_dim:
            root_dim = std_value_sort[-1]
        else:
            root_dim = std_value_sort[-2]
        return root_dim

    def __get_argsort_features_labels(self, features, labels, sort_dim):
        arg_indices = np.argsort(features[:, sort_dim])
        features_argsort = features[arg_indices]
        labels_argsort = labels[arg_indices]
        return features_argsort, labels_argsort

    def search(self, x, count=1):
        nearest = []
        for i in range(count):
            nearest.append([-1, None])
        self.nearest = np.array(nearest)

        def recurve(node):
            if node is not None:
                axis = node.depth % self.n
                daxis = x[axis] - node.data[axis]
                if daxis < 0:
                    recurve(node.left_child)
                else:
                    recurve(node.right_child)

                dist = np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(x, node.data)))
                for i, d in enumerate(self.nearest):
                    if d[0] < 0 or dist < d[0]:
                        self.nearest = np.insert(self.nearest, i, [dist, node], axis=0)
                        self.nearest = self.nearest[:-1]
                        break

                n = list(self.nearest[:, 0]).count(-1)
                if self.nearest[-n - 1, 0] > abs(daxis):
                    if daxis < 0:
                        recurve(node.right_child)
                    else:
                        recurve(node.left_child)

        recurve(self.KdTree)

        knn = self.nearest[:, 1]
        belong = []
        for i in knn:
            belong.append(i.data[-1])
        b = max(set(belong), key=belong.count)

        return self.nearest, b


kdtree = KdTree()
kdtree.create(features, labels)
ddd = 0
