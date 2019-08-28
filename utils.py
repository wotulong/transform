# -*- coding: utf-8 -*-
'''
2019-08-27
copy from:
https://www.github.com/kyubyong/transformer.
'''


import tensorflow as tf
import json
import os, re
import logging


logging.basicConfig(level=logging.INFO)


def calc_num_batches(total_num, batch_size):
    '''
    计算batch数量
    :param total_num: 样本总量
    :param batch_size:
    :return:
    '''

    return total_num // batch_size + int(total_num % batch_size != 0)


def convert_idx_to_token_tensor(inputs, idx2toeken):
    '''
    int32 tensor to string tensor
    :param inputs:
    :param idx2toeken:
    :return:
    '''

    def func(inputs):
        return " ".join(idx2toeken[elem] for elem in inputs)

    return tf.py_func(func, [inputs], tf.string)


def postprocess(hypotheses, idx2token):
    '''
    Process translationn outputs
    :param hypothess: list of encoded predictions
    :param idx2token: dictionary
    :return:
    '''

    _hypotheses = []
    for h in hypotheses:
        sent = "".join(idx2token[idx] for idx in h)
        sent = sent.split("</s>")[0].strip()
        sent = sent.replace("_", " ").strip()
        _hypotheses.append(sent)

    return _hypotheses


def save_hparams(hparams, path):
    '''
    保存超参数
    :param hparams:
    :param path:
    :return:
    '''

    os.makedirs(path, exist_ok=True)
    hp = json.dumps(vars(hparams))

    with open(os.path.join(path, "hparams"), 'w') as fout:
        fout.write(hp)


def load_hparams(parser, path):
    '''
    加载超参数
    :param parser: argsparse parser
    :param path:
    :return:
    '''

    if not os.path.isdir(path):
        path = os.path.dirname(path)

    with open(os.path.join(path, "hparams"), "r") as f:
        d = f.read()

    flagwval = json.loads(d)
    for f, v, in flagwval.items():
        parser.f = v


def save_variable_specs(fpath):
    '''
    保存变量信息
    :param fpath:
    :return:
    '''

    def _get_size(shape):
        '''
        获取shape大小
        :param shape:
        :return:
        '''
        size = 1
        for d in shape:
            size *= d

        return size

    params, num_params = [], 0
    for v in tf.global_variables():
        params.append("{}==={}".format(v.name, v.shape))
        num_params += _get_size(v.shape)
    print("num_params: %d" % num_params)

    with open(fpath, 'w') as fout:
        fout.write("num_params : %d\n" % num_params)
        fout.write("\n".join(params))

    logging.info("Variables's info has been saved.")


def get_hypootheses(num_batches, num_samples, sess, tensor, dict):
    '''
    获取超参数
    :param num_batches:
    :param num_samples:
    :param sess:
    :param tensor:
    :param dict:
    :return:
    '''

    hypotheses = []
    for _ in range(num_batches):
        h = sess.run(tensor)
        hypotheses.extend(h.tolist())
    hypotheses = postprocess(hypotheses, dict)

    return hypotheses[:num_samples]


def calc_bleu(ref, translation):
    '''
    计算bleu分数
    :param ref:
    :param translation:
    :return:
    '''

    get_bleu_score = "perl mullti-bleu.perl {} < {} > {}".format(ref, translation, "temp")
    os.system(get_bleu_score)
    with open(translation, 'r') as f:
        bleu_score_report = f.read()

    try:
        score = re.findall("BLEU = ([^,])+", bleu_score_report)[0]
        new_translation = translation + "B{}".format(score)
        os.system("mv {} {}".format(translation, new_translation))
        os.remove(translation)
    except:
        pass

    os.remove("temp")
