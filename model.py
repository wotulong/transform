#-*- coding: utf-8 -*-
'''
20190823 zsk
copy from:https://github.com/Kyubyong/transformer/blob/master/model.py
Transformer network
'''

import tensorflow as tf

from data_load import load_vocab
from modules import get_token_embeddings, ff, positional_encoding, multihead_attention, label_smoothing, noam_scheme
from utils import convert_idx_to_token_tensor
from tqdm import tqdm
import logging


logging.basicConfig(level=logging.INFO)


class Transformer:
    '''

    '''
    def __init__(self, hp):
        self.hp = hp
        self.token2idx, self.idx2token = load_vocab(hp.vocab)
        self.embeddings = get_token_embeddings(self.hp.vocab_size, self.hp.d_model, zero_pad=True)

    def encode(self, xs, training=True, scope="transformer_encoder"):
        '''
        编码器
        :param xs: (x, x_seqlens, sents1)
        :param training:boolean，是否进行dropout
        :return:
        '''

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            x, seqlens, sents1 = xs

            # embedding->[N, T1, d_model]
            enc = tf.nn.embedding_lookup(self.embeddings, x)
            enc *= self.hp.d_model ** 0.5

            enc += positional_encoding(enc, self.hp.maxlen1)
            enc = tf.layers.dropout(enc, self.hp.drouput_rate, training=training)


            # Blocks
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_block_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    enc = multihead_attention(
                        queries = enc,
                        keys = enc,
                        values = enc,
                        num_heads = self.hp.num_heads,
                        dropout_rate = self.hp.dropout_rate,
                        training = training,
                        causality = False
                    )
                    # feed forward
                    enc = ff(enc, num_units=[self.hp.d_ff, self.hp.d_model])
            memory = enc
            return memory, sents1

    def decode(self, ys, memory, training=True, scope="transformer_decoder"):
        '''
        解码器
        :param ys: (decoder_input, y, y_seqlens, sents2)
        :param memory:encoder outputs
        :param training:boolean
        :return:logits, y_hat, y, sents2
        '''

        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            decoder_inputs, y, seqlens, sents2 = ys

            # embedding
            dec = tf.nn.embedding_lookup(self.embeddings, decoder_inputs)
            dec *= self.hp.d_model ** 0.5

            dec += positional_encoding(dec, self.hp.maxlen2)
            dec = tf.layers.dropout(dec, self.hp.dropout_rate, training=training)

            # Blocks
            for i in range(self.hp.num_blocks):
                # Masked self-attention(在此处causality == True)
                dec = multihead_attention(
                    queries = dec,
                    keys = dec,
                    num_heads = self.hp.num_heads,
                    dropout_rate = self.hp.dropout_rate,
                    training = training,
                    causality = True,
                    scope = "self_attention"
                )

                # feed forward
                dec = ff(dec, num_units=[self.hp.d_ff, self.hp.d_model])

        # final linear projection
        weights = tf.transpose(self.embeddings) # [d_model, vocab_size]
        logits = tf.einsum('ntd,dk->ntk', dec, weights) # [N, T2, vocab_size]
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, sents2

    def train(self, xs, ys):
        '''
        模型训练
        :param xs: (x, x_seqlens, sents1)
        :param ys: (decoder_input, y, y_seqlens, sents2)
        :return:
        '''

        # forward
        memory, sents1 = self.encode(xs)
        logits, preds, y, sents2 = self.decode(ys, memory)

        # train scheme
        y_ = label_smoothing(tf.one_hot(y, depth=self.hp.vocab_size))
        ce = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_)
        nonpadding = tf.to_float(tf.not_equal(y, self.token2idx["<pad>"]))
        loss = tf.reduce_sum(ce * nonpadding) / (tf.reduce_sum(nonpadding) + 1e-8)

        global_step = tf.train.get_or_create_global_step
        lr = noam_scheme(self.hp.lr, global_step, self.warmup_step)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        # summary
        tf.summary.scalar('lr', lr)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('blobal_step', global_step)
        summaries = tf.summary.merge_all()

        return loss, train_op, global_step, summaries

    def eval(self, xs, ys):
        '''
        推理评估
        :param xs: (x, x_seqlens, sents1)
        :param ys: 忽略
        :return: y_hat->[N, T2]
        '''

        decoder_inputs, y, y_seqlen, sents2 = ys

        decoder_inputs = tf.ones((tf.shape(xs[0][0], 1), tf.int32) * self.token2idx["<s>"])
        ys = (decoder_inputs, y, y_seqlen, sents2)

        memory, sents1 = self.encode(xs, False)

        logging.info("Inference graph is being built, please be patient")
        for _ in tqdm(range(self.hp.maxlen2)):
            logits, y_hat, y, sents2 = self.decode(ys, memory, False)
            if tf.reduce_sum(y_hat, 1) == self.token2idx["<pad>"]:
                break

            _decoder_inputs = tf.concat((decoder_inputs, y_hat), 1)
            ys = (_decoder_inputs, y, y_seqlen, sents2)

        # 随机测试
        n = tf.random_uniform((), 0, tf.shape(y_hat)[0] - 1, tf.int32)
        sent1 = sents1[n]
        pred = convert_idx_to_token_tensor(y_hat[n], self.idx2token)
        sent2 = sents2[n]

        # summary
        tf.summary.text("sent1", sent1)
        tf.summary.text("pred", pred)
        tf.summary.text("sent2", sent2)
        summaries = tf.summary.merge_all()

        return y_hat, summaries

