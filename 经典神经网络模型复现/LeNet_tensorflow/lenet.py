# ----------------------------
#!  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------

import sys
import tensorflow as tf

from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
mnist = read_data_sets('MNIST_data', one_hot=True)


# ###官方提示最新下载方法，可是我这没有下载成功，还是用原来的方法
# from tensorflow.models.official.mnist import dataset
# mnist=dataset.download('./mnist','MNIST_DATA')

sys.setrecursionlimit(100000)  # 例如这里设置为十万


class LeNetBaseClass:
    def __init__(self, image_size=(28, 28, 1), class_num=10,
                 conv_filters_tuple=(6, 16), conv_filter_size_tuple=(5, 5),
                 pooling_size_tuple=(2, 2), pooling_size_stride=(2, 2),
                 fc_units_tuple=(120, 84), learning_rate=0.001,
                 activate_function=tf.nn.relu,
                 pooling_function=tf.layers.max_pooling2d):

        self._model(
            image_size,
            class_num,
            conv_filters_tuple,
            conv_filter_size_tuple,
            pooling_size_tuple,
            pooling_size_stride,
            fc_units_tuple,
            learning_rate,
            activate_function,
            pooling_function)
        self._sess = tf.Session()

    def train(self):
        raise NotImplementedError('必须在子类重写该方法')

    def evaluate(self):
        raise NotImplementedError('必须在子类重写该方法')

    def inference(self):
        raise NotImplementedError('必须在子类重写该方法')

    def _model(self, image_size, class_num,
               conv_filters_tuple, conv_filter_size_tuple,
               pooling_size_tuple, pooling_size_stride,
               fc_units_tuple, learning_rate,
               activate_function, pooling_function):
        image_height, image_width, image_channel = image_size
        input_feature = tf.placeholder(
            shape=[
                None,
                image_height * image_width
            ],
            dtype=tf.float32)
        input_feature_reshape = tf.reshape(
            input_feature, [-1, image_height, image_width,
                            image_channel])
        input_label = tf.placeholder(
            shape=[
                None,
                class_num],
            dtype=tf.float32)
        conv_layer1 = tf.layers.conv2d(
            input_feature_reshape,
            kernel_size=conv_filter_size_tuple[0],
            filters=conv_filters_tuple[0],
            activation=activate_function,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.1))
        pooling_layer1 = pooling_function(
            conv_layer1, pool_size=pooling_size_tuple[0],
            strides=pooling_size_stride[0])
        conv_layer2 = tf.layers.conv2d(
            pooling_layer1,
            kernel_size=conv_filter_size_tuple[1],
            filters=conv_filters_tuple[1],
            activation=activate_function,
            kernel_initializer=tf.truncated_normal_initializer(
                stddev=0.1))
        pooling_layer2 = pooling_function(
            conv_layer2, pool_size=pooling_size_tuple[1],
            strides=pooling_size_stride[1])
        pooling_layer2_flat = tf.layers.flatten(pooling_layer2)
        full_connection_layer1 = tf.layers.dense(
            pooling_layer2_flat, fc_units_tuple[0],
            activation=activate_function)
        full_connection_layer2 = tf.layers.dense(
            full_connection_layer1, fc_units_tuple[1],
            activation=activate_function)
        output_layer = tf.layers.dense(full_connection_layer2, class_num)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=output_layer, labels=input_label)
        predict_label = tf.argmax(tf.nn.softmax(output_layer), 1)
        correct_prediction = tf.equal(
            tf.argmax(input_label, 1), predict_label)
        accuracy = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))
        train_step = tf.train.AdamOptimizer(learning_rate).minimize(
            cross_entropy)
        self.train_io_parameters_dict = {
            'inputs': {
                'input_feature': input_feature,
                'input_label': input_label},
            'outputs': {
                'train_step': train_step,
                'accuracy': accuracy,
                'cross_entropy': cross_entropy}}
        self.eval_io_parameters_dict = {
            'inputs': {
                'input_feature': input_feature,
                'input_label': input_label},
            'outputs': {
                'accuracy': accuracy,
                'cross_entropy': cross_entropy}}

        self.inferece_io_parameters_dict = {
            'inputs': {
                'input_feature': input_feature,
                'input_label': input_label},
            'outputs': {
                'predict_label': predict_label}}


class LeNet(LeNetBaseClass):
    def train(self):
        input_feature = self.train_io_parameters_dict['inputs']['input_feature']
        input_label = self.train_io_parameters_dict['inputs']['input_label']
        accuracy = self.train_io_parameters_dict['outputs']['accuracy']
        train_step = self.train_io_parameters_dict['outputs']['train_step']
        self._sess.run(tf.global_variables_initializer())
        self._sess.run(tf.local_variables_initializer())
        for step in range(1000):
            batch = mnist.train.next_batch(60)
            if step % 100 == 0:
                accuracy_value = self._sess.run(accuracy, feed_dict={
                    input_feature:
                        batch[0],
                    input_label: batch[1]})
                print('训练第%d个step，精度%f \n' % (step, accuracy_value))
                tt = 0
            train_step.run(
                session=self._sess,
                feed_dict={
                    input_feature: batch[0],
                    input_label: batch[1]})

    def evaluate(self):
        input_feature = self.eval_io_parameters_dict['inputs']['input_feature']
        input_label = self.eval_io_parameters_dict['inputs']['input_label']
        accuracy = self.eval_io_parameters_dict['outputs']['accuracy']
        print(
            'test accuracy: {}'.format(
                accuracy.eval(
                    session=self._sess,
                    feed_dict={
                        input_feature: mnist.test.images,
                        input_label: mnist.test.labels})))

    def inference(self):
        input_feature = self.inferece_io_parameters_dict['inputs']['input_feature']
        predict_label_tensor = self.inferece_io_parameters_dict['outputs'][
            'predict_label']
        predict_label = self._sess.run(predict_label_tensor, feed_dict={
            input_feature: mnist.test.images})
        return predict_label


def main():
    lenet = LeNet()
    lenet.train()
    lenet.evaluate()
    predict = lenet.inference()


if __name__ == "__main__":
    main()
