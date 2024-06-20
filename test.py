#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 7:25 PM
# @Author  : sunwenjun
# @File    : test.py
# @brief: test

import os
from helper.read_pdf import read_pdf_ch_large
from helper.rag import build_prompt, get_completion
from helper.vecdb import VectorDBConnector
from helper.embedding import sents2embedding

def build_vecdb():



    return


def test():
    # 1、读取pdf文档
    pdf_filepath = './data/基于LSTM和启发式搜索的遥感卫星地面站天线智能调度方法研究2020.pdf'  # 示例PDF文件路径
    text_list = read_pdf_ch_large(pdf_filepath)  # 读取PDF文件中的文本
    print("text_list:", len(text_list))

    # 提取文件名
    filename = os.path.basename(pdf_filepath)

    # 2、建立矢量数据库
    vec_db = VectorDBConnector(collection_name="paper_collection", embedding_fn=sents2embedding)
    # vec_db.add_document(text_list)

    # 3、Prompt模版
    prompt_template = """
    已知信息:
    __INFO__
    
    用户问：
    __QUERY__
    
    请用中文回答用户问题。
    """

    print("prompt_template", prompt_template)

    # 4、检索
    # user_query = "LSTM网络是如何应用在地面站天线调度的？"
    # user_query = "论文中使用的LSTM是单向的还是双向的？"
    # user_query = "论文中使用的LSTM是一层的还是两层的？"
    # user_query = "这篇论文的作者是谁？"
    user_query = "这篇论文中作者致谢了哪些人？"
    search_results = vec_db.search(user_query, 10)
    res_text_list = search_results['documents'][0]

    # 5、根据Prompt模版生成Prompt
    prompt = build_prompt(prompt_template, info=res_text_list, query=user_query)
    print("Prompt:\n", prompt)

    # 6、调用大模型,参考检索结果，生成问题答案
    response = get_completion(prompt)
    print("答案：\n", response)

    return response


test()
