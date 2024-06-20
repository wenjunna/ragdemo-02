#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 6:51 PM
# @Author  : sunwenjun
# @File    : embedding.py
# @brief: 智源的这个中文模型表现挺好，维度是1024维。真正上线的时候，把模型提前加载到内存中

from sentence_transformers import SentenceTransformer, util
from appconf import embedding_model_path


def sents2embedding(sentences):
    '''
    使用sentence-transformers库加载模型,获取句子向量
    :param sentences: 句子列表 []
    :return: 句子向量
    '''

    # model_name = "princeton-nlp/sup-simcse-bert-base-uncased"  # 也可以使用其他预训练模型，如 unsup-simcse-bert-base-uncased
    # model = SentenceTransformer(model_name)

    # 文本转向量
    model = SentenceTransformer(embedding_model_path)  # 换成本地模型存放路径

    # 生成句子嵌入
    embeddings = model.encode(sentences, convert_to_tensor=True)
    print(embeddings.shape)  # torch.Size([6, 1024])
    embeddings = embeddings.numpy()

    return embeddings


if __name__ == '__main__':
    # 示例句子
    # sentences = ["This is a sentence.", "art is wonderful"]
    sentences = ["NLP算法工程师", "自然语言处理算法工程师", "计算机视觉算法工程师", "大模型算法工程师", "JAVA开发", "平面设计师"]
    embeddings = sents2embedding(sentences)

    # 计算句子之间的余弦相似性
    cosine_similarities = util.pytorch_cos_sim(embeddings, embeddings)
    print(cosine_similarities)
