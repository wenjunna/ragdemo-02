#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 7:00 PM
# @Author  : sunwenjun
# @File    : vecdb.py
# @brief: PyCharm

import chromadb
from chromadb.config import Settings
from helper.simcse_embedding import get_embedding
from helper.read_pdf import read_pdf


class VectorDBConnector(object):
    def __init__(self, collection_name, embedding_fn):
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        chroma_client.reset()

        # 创建一个collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        self.embedding_fn = embedding_fn

    def add_document(self, documents, metadata={}):
        '''
        向collection中添加文档与向量
        :param documents:
        :param metadata:
        :return:
        '''
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 每个文档的向量
            documents=documents,  # 原文档
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的id
        )

    def search(self, query, top_n):
        '''
        检索向量数据库
        :param query:
        :param top_n:
        :return:
        '''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results


if __name__ == '__main__':
    # pdf_filepath = '../data/孙文军-NLP算法岗-4年-3.pdf'  # 示例PDF文件路径
    pdf_filepath = '../data/基于LSTM和启发式搜索的遥感卫星地面站天线智能调度方法研究2020.pdf'  # 示例PDF文件路径
    pdf_text = read_pdf(pdf_filepath, min_line_length=5)  # 读取PDF文件中的文本
    print("pdf_text:", len(pdf_text))

    vec_db = VectorDBConnector("demo", get_embedding)
    vec_db.add_document(pdf_text)

    # user_query = "孙文军是那一年出生的？"
    user_query = "这篇文章使用的LSTM网络是几层？"
    results = vec_db.search(user_query, 5)

    for para in results['documents'][0]:
        print(para + "\n")