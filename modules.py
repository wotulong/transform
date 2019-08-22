#-*- coding: utf-8 -*-

'''
20190822
copy from:
https://github.com/Kyubyong/transformer/blob/master/modules.py

building block for Transformer
'''


import numpy as np
import tensorflow as tf


def ln(inputs, epsilon = 1e-8, scope='ln'):
    '''
    Applice layer normalization.https://arxiv.org/abs/1607.06450
    :param inputs: A tensor [B,...]
    :param epsilon:A floating number. 防止除0
    :param scope:变量域
    :return:layer norm后的值
    '''

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initialize())
        normalized = (inputs - mean) / ((epsilon + variance) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def get_token_embeddings(vocab_size, num_units, zero_pad=True):
    '''
    构造Token的embedding，index 0 设置为0
    :param vocab_size:
    :param num_units: embedding size
    :param zero_pad: Boolean, 是否用0补齐
    :return:
    '''

    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable(
            'weight_mat',
            dtype = tf.float32,
            shape = (vocab_size, num_units),
            initializer = tf.contrib.layers.xavier_initializer()
        )

        # 首列补0
        tf.zero_pad:
            embedding = tf.concat((tf.zeros(shape=[1, num_units]),
                                   embeddings[1:, :]), 0)

    return embeddings