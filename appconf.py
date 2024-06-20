#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 7:42 PM
# @Author  : sunwenjun
# @File    : appconf.py
# @brief: 参数配置文件

stopwords_path = './dict/stopwords.dict'

# 文本转向量的模型
# embedding_model_path = "/Users/sunwenjun/data/models_from_hf/princeton-nlp/sup-simcse-bert-base-uncased"
embedding_model_path = "/Users/sunwenjun/data/models_from_hf/BAAI/bge-large-zh-v1.5"

# 向量数据库路径
chromadb_path = "/Users/sunwenjun/data/chromadb_path"

# collection name
collection_name = "paper_collection"
