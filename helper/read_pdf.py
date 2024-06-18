#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/13 8:55 AM
# @Author  : sunwenjun
# @File    : read_pdf2.py
# @brief: PyCharm
import pdfplumber

from helper.para_sent_seg import ParaSentSeg
from helper.str_qj2bj import clean_norm


def read_pdf(filepath, min_line_length):
    '''
    取PDF文件并提取文本
    :param filepath: 文件路径
    :param min_line_length: 文本长度低于该值，就分段
    :return: []
    '''
    paragraphs = []
    buffer = ''
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            full_text = page.extract_text()
            # 按空行分隔，将文本重新组织成段落
            lines = full_text.split('\n')
            for text in lines:
                if len(text) >= min_line_length:
                    buffer += (' ' + text) if not text.endswith('-') else text.strip('-')
                elif buffer:
                    paragraphs.append(buffer)
                    buffer = text
            if buffer:
                paragraphs.append(buffer)
    return paragraphs


def preprocess(text_list):
    results = []
    for text in text_list:
        text = clean_norm(text)
        stopwords_file = "../dict/stopwords.txt"
        psc = ParaSentSeg(stopwords_file=stopwords_file)
        sents_list = psc.para_seg(text)
        for sents in sents_list:
            results.append(sents)

    return results


if __name__ == '__main__':
    # 示例PDF文件路径
    pdf_filepath = '../data/基于LSTM和启发式搜索的遥感卫星地面站天线智能调度方法研究2020.pdf'

    # 读取PDF文件中的文本
    text_list = read_pdf(pdf_filepath, min_line_length=10)
    # for text in text_list:
    #     print(text)

    results = preprocess(text_list)
    for text in results:
        print(text)