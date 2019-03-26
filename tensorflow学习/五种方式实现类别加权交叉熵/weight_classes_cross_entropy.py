# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：五种方式实现类别加权交叉熵
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------

"""
标签为１的类别权重变为其他类别的１０倍
"""


def weight_classes_cross_entropy_python():
    import numpy as np

    def softmax(x):
        sum_raw = np.sum(np.exp(x), axis=-1)
        x1 = np.ones(np.shape(x))
        for i in range(np.shape(x)[0]):
            x1[i] = np.exp(x[i]) / sum_raw[i]
        return x1

    logits = np.array([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]])
    labels = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]])  # 每一行只有一个1
    coe = [1, 10, 1, 1, 10]
    logits_softmax = softmax(logits)
    cross_entropy_vector = np.sum(-labels * np.log(logits_softmax), axis=1)
    cross_entropy = np.mean(cross_entropy_vector * coe)

    print('weight_classes_cross_entropy_python计算结果：%5.4f' % cross_entropy)


def weight_classes_cross_entropy_tf_losess():
    import tensorflow as tf
    labels = tf.Variable(initial_value=tf.constant([0, 1, 2, 0, 1]), dtype=tf.int32)
    logits = tf.Variable(initial_value=tf.constant([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]]), dtype=tf.float32)
    coe = tf.where(tf.equal(labels, 1), tf.multiply(10, tf.ones_like(labels)), tf.ones_like(labels))
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels, weights=coe)
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        cross_entropy_value = sess.run(cross_entropy)
    print('weight_classes_cross_entropy_python_tf_losess计算结果：%5.4f' % cross_entropy_value)


def weight_classes_cross_entropy_tf_nn_sparse():
    import tensorflow as tf
    labels = tf.Variable(initial_value=tf.constant([0, 1, 2, 0, 1]), dtype=tf.int32)
    logits = tf.Variable(initial_value=tf.constant([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]]), dtype=tf.float32)
    coe = tf.where(tf.equal(labels, 1), tf.multiply(tf.constant(10, dtype=tf.float32), tf.ones_like(labels, dtype=tf.float32)), tf.ones_like(labels, dtype=tf.float32))
    cross_entropy_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)

    cross_entropy = tf.reduce_mean(tf.multiply(coe, cross_entropy_vector))
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        cross_entropy_value = sess.run(cross_entropy)

    print('weight_classes_cross_entropy_python_tf_nn_sparse计算结果：%5.4f' % cross_entropy_value)


def weight_classes_cross_entropy_tf_nn():
    import tensorflow as tf
    onehot_labels = tf.Variable(initial_value=tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]), dtype=tf.int32)
    labels = tf.arg_max(onehot_labels, 1)
    logits = tf.Variable(initial_value=tf.constant([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]]), dtype=tf.float32)
    coe = tf.where(tf.equal(labels, 1), tf.multiply(tf.constant(10, dtype=tf.float32), tf.ones_like(labels, dtype=tf.float32)), tf.ones_like(labels, dtype=tf.float32))
    cross_entropy_vector = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=logits)

    cross_entropy = tf.reduce_mean(tf.multiply(coe, cross_entropy_vector))
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        cross_entropy_value = sess.run(cross_entropy)
    print('weight_classes_cross_entropy_python_tf_nn计算结果：%5.4f' % cross_entropy_value)


def weight_classes_cross_entropy_tf():
    import tensorflow as tf
    onehot_labels = tf.Variable(initial_value=tf.constant([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]), dtype=tf.float32)
    labels = tf.arg_max(onehot_labels, 1)
    logits = tf.Variable(initial_value=tf.constant([[12, 3, 2], [3, 10, 1], [1, 2, 5], [4, 6.5, 1.2], [3, 6, 1]]), dtype=tf.float32)
    labels_softmax = tf.nn.softmax(logits)
    cross_entropy_vector = -tf.reduce_sum(tf.multiply(onehot_labels, tf.log(labels_softmax)), axis=1)
    coe = tf.where(tf.equal(labels, 1), tf.multiply(tf.constant(10, dtype=tf.float32), tf.ones_like(labels, dtype=tf.float32)), tf.ones_like(labels, dtype=tf.float32))
    cross_entropy = tf.reduce_mean(tf.multiply(coe, cross_entropy_vector))
    with tf.Session()as sess:
        sess.run(tf.global_variables_initializer())
        cross_entropy_value = sess.run(cross_entropy)
    print('weight_classes_cross_entropy_tf计算结果：%5.4f' % cross_entropy_value)


if __name__ == '__main__':
    weight_classes_cross_entropy_python()
    weight_classes_cross_entropy_tf_losess()
    weight_classes_cross_entropy_tf_nn_sparse()
    weight_classes_cross_entropy_tf_nn()
    weight_classes_cross_entropy_tf()
