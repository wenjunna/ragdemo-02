#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/3 11:24 PM
# @Author  : sunwenjun
# @File    : para_sent_seg.py
# @brief: 使用jieba、正则表达式实现分句、分词

import re
import jieba


class ParaSentSeg(object):
    def __init__(self, stopwords_file='./stopwords.txt', stopwords=False):
        self.stopwords = stopwords
        self.stopwords_list = []
        if self.stopwords:
            self.stopwords_list = self.load_stop_words(stopwords_file)

    def load_stop_words(self, stop_file):
        print("load stop words start")

        data_list = []
        line_cnt = 0

        file = open(stop_file, 'r', encoding='utf8')
        while True:
            line_cnt += 1
            line = ""

            try:
                line = file.readline()
            except:
                print("error line: {}".format(line))
                line = "error line"

            if not line:
                break

            line = line.strip().rstrip("\n")
            data_list.append(line)

        print("load stop words done, data total = %d, line count = %d" % (len(data), line_cnt))

        return data_list

    def sentence_seg(self, txt):
        '''
        分词
        :param txt: str  "参与并完成多个项目渗透测试"
        :return: list   ['参与', '完成', '多个', '项目', '渗透', '测试']
        '''

        # 分词
        seg_list = []
        words = jieba.cut(txt)
        for word in list(words):
            word = word.strip()
            # 过滤掉停用词、分词后长度小于等于1的词
            if len(word) < 2 or (word in self.stopwords_list) or word.isspace():
                continue
            seg_list.append(word)

        return seg_list

    def para_seg(self, txt):
        '''
        分句,一个长的段落拆分成几个短的句子
        :param txt: str  "1.开发与维护合作商户关系 2.驻点办理分期业务 3.商户服务费及销售数据对账结算工作"
        :return: list   ["开发与维护合作商户关系","驻点办理分期业务","商户服务费及销售数据对账结算工作"]
        '''
        sent_list = []

        # 找出断句标识
        pattern = re.compile("([【（[][0123456789]{1,2}[]】）])|"
                             "([【（[]\w{1,6}[]】）])|"
                             "([\s,，。.]+\w{1,6}：)|"
                             "([\s,，。.]+[0123456789]{1,2}[、.，)）])|"
                             "([。；;!！？?<>])|"
                             "(^[0123456789]{1,2}[、.，)）])")

        if re.findall(pattern, txt):
            para = pattern.sub('##', txt)
            sents = para.split('##')
            for sent in sents:
                if 1 < len(sent.strip()):
                    sent_list.append(sent.strip())

        if len(sent_list) == 0:
            return [txt]

        return sent_list


if __name__ == '__main__':
    stopwords_file = "../dict/stopwords.txt"
    psc = ParaSentSeg(stopwords_file=stopwords_file)

    # 分词测试
    sent = "参与并完成多个项目渗透测试"
    words_list = psc.sentence_seg(sent)
    print(words_list)  # ['参与', '完成', '多个', '项目', '渗透', '测试']

    # 分句测试
    para = "1.开发与维护合作商户关系 2.驻点办理分期业务 3.商户服务费及销售数据对账结算工作"
    sents_list = psc.para_seg(para)
    print(sents_list)  # ['开发与维护合作商户关系', '驻点办理分期业务', '商户服务费及销售数据对账结算工作']
