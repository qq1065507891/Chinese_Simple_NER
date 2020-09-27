#!/usr/bin/python
# -*- coding: UTF-8 -*-
# Author 曾怡霖

import codecs
import data_utils


def load_sentences(path):
    """
    加载数据集, 每一行至少包含一个汉字和一个标记
    句子和句子之间以空格隔开
    最后返回句子集合
    :param path:
    :return:
    """
    # 句子集合
    sentences = []
    # 临时存放句子的集合
    sentence = []
    with codecs.open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            # 判断句子是否为空, 如果为空则为句子与句子之间的的分割点
            if not line:
                # 判断句子集合是否为空, 如果不为空则加入句子集合中
                if len(sentence) > 0:
                    sentences.append(sentence)
                    # 清空临时句子列表
                    sentence = []
            else:
                if line[0] == ' ':
                    continue
                else:
                    word = line.split()
                    assert len(word) == 2
                    sentence.append(word)
        # 循环走完, 判断临时句子集合中是否还有句子, 有则加入句子集合
        if len(sentence) > 0:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    将BIO编码更新为BIOES编码
    :param sentences:
    :param tag_scheme:
    :return:
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        if not data_utils.check_bio(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('输入的句子应为BIO编码, 请检查句子%i:\n%s' % (i, s_str))

        if tag_scheme == 'BIO':
            for word, tag in zip(s, tags):
                word[-1] = tag

        elif tag_scheme == 'BIOES':
            new_tags = data_utils.bio_to_bioes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag

        else:
            raise Exception('非法编码')


def word_mapping(sentences):
    """
    构建字典
    :param sentences:
    :return:
    """
    word_list = [[x[0] for x in s]for s in sentences]
    dico = data_utils.create_dico(word_list)
    dico['<PAD>'] = 10000001
    dico['<UNK>'] = 10000000
    word_to_id, id_to_word = data_utils.create_mapping(dico)
    return dico, word_to_id, id_to_word


def tag_mapping(sentences):
    """
    构建标签字典
    :param sentences:
    :return:
    """
    tag_list = [[x[1] for x in s]for s in sentences]
    dico = data_utils.create_dico(tag_list)
    tag_to_id, id_to_tag = data_utils.create_mapping(dico)
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, word_to_id, tag_to_id, train=True):
    """
    数据预处理, 返回list其中包括:
    -word_list
    -word_id_list
    -word char index
    -tag_id_List
    :param sentences:
    :param word_to_id:
    :param tag_to_id:
    :param train:
    :return:
    """
    none_to_id = tag_to_id['O']

    data = []
    for s in sentences:
        word_list = [w[0] for w in s]
        word_id_list = [word_to_id[x if x in word_to_id else '<UNK>'] for x in word_list]
        segs = data_utils.get_seg_features(''.join(word_list))

        if train:
            tag_id_list = [tag_to_id[w[-1]] for w in s]
        else:
            tag_id_list = [none_to_id for _ in s]
        data.append([word_list, word_id_list, segs, tag_id_list])

    return data


if __name__ == '__main__':
    path = './data/ner.dev'
    sentences = load_sentences(path)
    update_tag_scheme(sentences, 'BIOES')
    # print('over')
    _, word_to_id, id_to_word = word_mapping(sentences)
    _, tag_to_id, id_to_tag = tag_mapping(sentences)
    dev_data = prepare_dataset(sentences, word_to_id, tag_to_id, train=True)
    data_utils.BatchManager(dev_data, 120)