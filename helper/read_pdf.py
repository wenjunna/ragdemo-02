#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/13 8:55 AM
# @Author  : sunwenjun
# @File    : read_pdf.py
# @brief: 读取pdf文档

import pdfplumber
from helper.para_sent_seg import ParaSentSeg
from helper.str_qj2bj import clean_norm
from appconf import stopwords_path

psc = ParaSentSeg(stopwords_file=stopwords_path)


def read_pdf_en(filepath):
    '''
    读取英文论文 PDF文件并提取文本
    :param filepath: 文件路径
    :return: []
    '''
    text_list = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            page_text_list = page_text.split("\n")
            text_list += page_text_list

    return text_list


def read_pdf_ch_large(filepath):
    '''
    中文大论文pdf解析
    每篇论文，句子长度分析。
    :param filepath: 文件路径
    :param min_line_length: 文本长度低于该值，就分段
    :return: []
    '''
    text_list = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            page_text_list = page_text.split("\n")
            text_list += page_text_list

    # 先使用句号分句
    text_list = " ".join(text_list).split("。")

    # 在进一步预处理
    results = []
    for text in text_list:
        text = clean_norm(text)
        if len(text) < 100:
            results.append(text)
            continue

        # 对于长度大于100的情况，进行进一步处理
        sents_list2 = psc.para_seg2(text)
        for sents in sents_list2:
            if len(sents) < 100:
                results.append(sents)
                continue

            # 对于长度大于100的情况，进行进一步处理
            sents_list3 = psc.para_seg3(sents)
            for sents in sents_list3:
                results.append(sents)

    return results


def read_pdf_ch_small(filepath):
    '''
    中文小论文pdf解析
    中文下论文一般会分栏，怎么处理分栏的情况
    每篇论文，句子长度分析。
    :param filepath: 文件路径
    :param min_line_length: 文本长度低于该值，就分段
    :return: []
    '''
    article_text_list = []

    return article_text_list


if __name__ == '__main__':
    # 示例PDF文件路径
    pdf_filepath = '../data/基于LSTM和启发式搜索的遥感卫星地面站天线智能调度方法研究2020.pdf'
    text_list = read_pdf_ch_large(pdf_filepath)
    for idx, text in enumerate(text_list):
        print('idx,长度，text:', idx, len(text), text)
