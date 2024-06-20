#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 7:00 PM
# @Author  : sunwenjun
# @File    : vecdb.py
# @brief: 向量数据库

import os
import chromadb
from chromadb.config import Settings
from helper.read_pdf import read_pdf_ch_large
from helper.embedding import sents2embedding
from appconf import chromadb_path


class VectorDBConnector(object):
    def __init__(self, collection_name, embedding_fn):
        # 第一种方式，基于内存
        # chroma_client = chromadb.Client(Settings(allow_reset=True))
        # chroma_client.reset()

        # 第二种方式，放在服务器   chroma run --path /Users/sunwenjun/data/chromadb_path
        # chroma_client = chromadb.HttpClient(host='localhost', port=8000, settings=Settings(allow_reset=False))

        # 第三种方式，持久化本地
        chroma_client = chromadb.PersistentClient(path=chromadb_path,
                                                  settings=Settings(allow_reset=True))

        # 创建一个collection  可以创建多个collection
        # chroma_client.delete_collection('my_collection')  # 删除collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)  # 获取或者新建collection
        self.embedding_fn = embedding_fn

    def add_document(self, documents):
        '''
        向collection中添加文档与向量
        :param documents: []
        :return:
        '''
        # 添加
        # self.collection.add(
        #     embeddings=self.embedding_fn(documents),  # 文档向量
        #     documents=documents,  # 原文档
        #     metadatas=metadatas,  # 元数据
        #     ids=ids  # 文档id
        # )
        #
        # 先查询，如果id不存在添加


        metadatas = []
        ids = []
        for i, text in enumerate(text_list):
            metadatas.append({'filename': filename, 'text_len': len(text)})
            ids.append(f"{filename}_id{i}")



        # 增加或者更新，如果原来就有，就覆盖，原来没有的就添加
        self.collection.upsert(
            embeddings=self.embedding_fn(documents),  # 文档向量
            documents=documents,  # 原文档
            metadatas=metadatas,  # 元数据
            ids=ids  # 文档id
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
    # 示例PDF文件路径
    pdf_filepath = '../data/基于LSTM和启发式搜索的遥感卫星地面站天线智能调度方法研究2020.pdf'
    filename = os.path.basename(pdf_filepath)

    # 读取PDF文件中的文本
    text_list = read_pdf_ch_large(pdf_filepath)
    print("text_list:", len(text_list))

    # 创建矢量数据库
    vec_db = VectorDBConnector("paper_collection", sents2embedding)
    vec_db.add_document(text_list)

    # 查询
    user_query = "这篇文章使用的LSTM网络是几层？"
    results = vec_db.search(user_query, 10)
    print("results", results)
    # res_text_list = results['documents'][0]
    #
    # for para in res_text_list:
    #     print(para + "\n")
