#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/18 7:25 PM
# @Author  : sunwenjun
# @File    : test_vec.py
# @brief: PyCharm

from helper.read_pdf import read_pdf,preprocess
from helper.rag import build_prompt, get_completion
from helper.vecdb import VectorDBConnector
from helper.simcse_embedding import get_embedding

# pdf_filepath = './data/孙文军-NLP算法岗-4年-3.pdf'  # 示例PDF文件路径
pdf_filepath = './data/基于LSTM和启发式搜索的遥感卫星地面站天线智能调度方法研究2020.pdf'  # 示例PDF文件路径
text_list = read_pdf(pdf_filepath, min_line_length=5)  # 读取PDF文件中的文本
print("text_list:", len(text_list))

pdf_text_list = preprocess(text_list)
for text in pdf_text_list:
    print(text)

vec_db = VectorDBConnector("demo", get_embedding)
vec_db.add_document(pdf_text_list)

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
# user_query = "孙文军是那一年出生的？"
user_query = "LSTM网络是如何应用在地面站天线调度的？"
search_results = vec_db.search(user_query, 10)

for para in search_results['documents'][0]:
    print(para + "\n")


# 5、根据Prompt模版生成Prompt
prompt = build_prompt(prompt_template, info=search_results['documents'][0], query=user_query)
print("Prompt:\n", prompt)

# 6、调用大模型,参考检索结果，生成问题答案
response = get_completion(prompt)
print("答案：\n", response)
