#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 6:51 PM
# @Author  : sunwenjun
# @File    : simcse_embedding.py
# @brief: PyCharm


from sentence_transformers import SentenceTransformer, util


def get_embedding(sentences):
    '''
    获取句子向量
    :param sentences: 句子列表
    :return: 句子向量
    '''

    # 使用sentence-transformers库加载模型

    # model_name = "princeton-nlp/sup-simcse-bert-base-uncased"  # 也可以使用其他预训练模型，如 unsup-simcse-bert-base-uncased
    # model = SentenceTransformer(model_name)

    local_model_path = "/Users/sunwenjun/data/models_from_hf/princeton-nlp/sup-simcse-bert-base-uncased"
    model = SentenceTransformer(local_model_path)  # 换成本地模型存放路径

    # 生成句子嵌入
    embeddings = model.encode(sentences, convert_to_tensor=True)
    # print(embeddings.shape)  # torch.Size([6, 768])
    embeddings = embeddings.numpy()

    return embeddings


if __name__ == '__main__':
    # 示例句子
    # sentences = ["This is a sentence.", "This is another sentence."]
    sentences = ["NLP算法工程师", "自然语言处理算法工程师", "计算机视觉算法工程师", "大模型算法工程师", "JAVA开发", "平面设计师"]
    embeddings = get_embedding(sentences)

    # 计算句子之间的余弦相似性
    cosine_similarities = util.pytorch_cos_sim(embeddings, embeddings)
    print(cosine_similarities)
