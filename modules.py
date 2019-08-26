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


def scaled_dot_product_attention(Q, K, V,
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''
    attention实现
    文章 3.2.1节

    :param Q: queries，3d tensor->[N, T_q, d_K]
    :param K: kyes, 3d tensor->[N, T_k, d_k]
    :param V: values, 3d tensor->[N, T_k, d_v]
    :param causality: if True, 应用masking
    :param dropout_rate: [0,1]浮点数
    :param training: 是否使用dropout
    :param scope: 域
    :return:
    '''

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # 点乘->[N, T_q, T_k]
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))

        # scale
        outputs /= d_k ** 0.5

        # key masking
        outputs = mask(outputs, Q, K, type="key")

        # masking
        if causality:
            outputs = mask(outputs, type='future')

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attentionn", tf.expand_dims(attention[:1], -1))


        # qury masking
        outputs = mask(outputs, Q, K, type="query")


        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)


        # weighted sum(context vectors)->[N, T_q, d_v]
        outputs = tf.matmul(outputs, V)

    return outputs


def mask(inputs, queries=None, keys=None, type=None):
    '''
    Masks，生成和inputs一样维度的对应的queries或者keys的mask，
    维度和inputs一样，非0为1，0为0，然后乘以inputs生成outputs.

    For example:
    ```
    >> queries = tf.constant([[ [1.],
                                [2.],
                                [0.]]], tf.float32) # (1, 3, 1)
    >> keys =    tf.constant([[[4.],
                               [0.]]], tf.float32)  # (1, 2, 1)
    >> inputs =  tf.constant([[[4., 0.],
                               [8., 0.],
                               [0., 0.]]], tf.float32)

    >> mask(inputs, queries, keys, "key")
    array([[[ 4.0000000e+00, -4.2949673e+09],
            [ 8.0000000e+00, -4.2949673e+09],
            [ 0.0000000e+00, -4.2949673e+09]]], dtype=float32)
    ```
    :param inputs: 3d tensor->[N, T_q, T_k]
    :param queries: 3d tensor->[N, T_q, d]
    :param keys: 3d tensor->[N, T_k, d]
    :param type: mask 类型（“query”,"key","value"）
    :return:
    '''

    inputs_shape = tf.shape(inputs)
    padding_num = -2 ** 32 + 1

    # [N, ?, ?] -> [N, T_q, T_k]
    if type in("k", "key", "keys"):
        # 生成masks
        masks = tf.sign(tf.reduce_sum(tf.abs(keys), axis=-1)) #[N, T_k]
        masks = tf.expand_dims(masks, 1) # [N, 1, T_k]
        masks = tf.tile(masks, [1,inputs_shape[1], 1]) # [N, T_q, T_k]

        # mask应用于inputs生成outputs
        outputs = inputs * masks

    elif type in("q","query","queries"):
        masks = tf.sign(tf.reduce_sum(tf.abs(queries), axis=-1))
        masks = tf.expand_dims(masks, axis=-1)
        masks = tf.tile(masks, [1, 1, inputs_shape[1]])

        outputs = inputs * masks

    # 其他情况
    elif type in("f","future","right"):
        diag_vals = tf.ones_like(inputs[0, :, :,])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        masks = tf.tile(tf.expand_dims(tril, 0), [inputs_shape[0], 1, 1])

        paddings = tf.ones_like(mask) * padding_num
        outputs = tf.where(tf.equal(masks, 0), paddings, inputs)

    else:
        print("Type error! Please input type correctly!")


    return outputs


def multihead_attention(queries, keys, values,
                        num_heads = 8,
                        dropoout_rate = 0.,
                        training = True,
                        causality = False,
                        scope = 'multihead_attention'):
    '''
    多头注意力机制，文章 3.2.2
    :param queries: [N, T_q, d_model]
    :param keys: [N, T_k, d_model]
    :param valluse: [N, T_k, d_model]
    :param num_heads:Number of heads
    :param dropoout_rate: float belongs to [0,1]
    :param training: if using dropout
    :param causality: if using mask
    :param scope:
    :return:
    '''

    d_model = queries.get_shape()[-1]


    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        # [N, ?, ?] -> [N, ?, d_model]
        Q = tf.layers.dense(queries, d_model, use_bias=False) # [N, T_q, d_model]
        K = tf.layers.dense(keys, d_model, use_bias=False)  # [N, T_k, d_model]
        V = tf.layers.dense(values, d_model, use_bias=False)  # [N, T_q, d_model]

        # Split and concat
        # [N, ?, d_model] -> [num_heads*N, ?, d_model/num_heads]
        Q_ = tf.conncat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.conncat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.conncat(tf.split(V, num_heads, axis=2), axis=0)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropoout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.spllit(outputs, num_heads, axis=0), axis=2)

        # Residual
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs


def ff(inputs, num_units, scope='positionwise_feedforward'):
    '''
    position-wise feed forward net, 文章3.3
    :param inputs: [N, T, C]
    :param num_units:
    :param scope:
    :return:
    '''

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs = ln(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''
    标签平滑,让0，1标签变为略大于0，略小于1的值
    See 5.4 and https://arxiv.org/abs/1512.00567.

    For example:
    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[
                                    [0, 0, 1],
                                    [0, 1, 0],
                                    [1, 0, 0]],
                                   [
                                    [1, 0, 0],
                                    [1, 0, 0],
                                    [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)
    >>
    [array([[
              [ 0.03333334,  0.03333334,  0.93333334],
              [ 0.03333334,  0.93333334,  0.03333334],
              [ 0.93333334,  0.03333334,  0.03333334]],
            [
              [ 0.93333334,  0.03333334,  0.03333334],
              [ 0.93333334,  0.03333334,  0.03333334],
              [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]
    ```
    :param inputs:[N, T, V], V is number of vocabulary
    :param epsilon:Smoothing rate
    :return:
    '''

    V = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / V)


def positional_encoding(inputs, maxlen,
                        masking = True,
                        scope = 'positional_encoding'):
    '''
    Sinusoidal positional encoding. 3.5
    :param inputs: [N, T, E]
    :param maxlen:scalar >=T
    :param masking: Boolean, if True, padding positions are set to zeros
    :param scope:
    :return:
    '''
    E = inputs.get_shape().as_list()[-1] # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1] # dynamic

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1]) #[N, T]

        # First part of PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)
        ])

        # Second part, apply the cosine to even columns and sin to odds
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2]) # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2]) # dim 2i + 1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32) # [maxlen, E]

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)


def noam_scheme(init_lr, global_step, warmup_steps=4000.):
    '''
    Noam scheme learning rate decay
    :param init_lr: scalar, initial learning rate
    :param gloabal_step: scalar
    :param warmup_steps: number of steps learning rate increase
    :return:
    '''

    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

