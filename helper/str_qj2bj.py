#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 10:48 PM
# @Author  : sunwenjun
# @File    : str_qj2bj.py
# @brief: 全角转半角，包括中英文括号、全半角

def strQJ2BJ(ustring):
    '''
    全角转半角
    '''
    rstring = ""

    for uchar in ustring:
        u_len = len(uchar.encode('unicode_escape'))
        # u_code = uchar.encode('unicode_escape')
        tmp = uchar.encode('unicode_escape')
        if u_len > 8:
            continue

        # 过滤干扰符号
        # ranges = "00010000-0010ffff;0080-00FF;D800-DBFF;DC00-DFFF;E000-F8FF;F900-FAFF;FB00-FB4F;FB50-FDFF;FE00-FE0F;FE10-FE1F;FE20-FE2F;FE30-FE4F;FE50-FE6F;FE70-FEFF;FFF0-FFFF;2600-26FF;2000-206F"
        ranges = "0000-007E;2E80-2EFF;3000-303F;31C0-31EF;3200-32FF;3300-33FF;3400-4DBF;4DC0-4DFF;4E00-9FBF;A700-A71F;AC00-D7AF;F900-FAFF;FE30-FE4F;FF00-FFEF;FF00-FFEF"

        ranges = ranges.strip().split(";")
        flag = 0
        for r in ranges:
            compare = r.split("-")
            if int(compare[0], 16) <= ord(uchar) and ord(uchar) <= int(compare[1], 16):
                flag = 1
                continue

        if flag == 0:
            continue

        inside_code = ord(uchar)

        SPACE_32 = 32
        SPACE_12288 = 12288
        SPACE_160 = 160
        SPACE_8194 = 8194
        SPACE_8195 = 8195
        SPACE_8197 = 8197
        SPACE_8201 = 8201

        if inside_code in [SPACE_32, SPACE_12288, SPACE_160, SPACE_8194, SPACE_8195, SPACE_8197,
                           SPACE_8201]:  # 全角空格直接转换
            inside_code = 32

        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring


def clean_norm(text):
    # text = text.strip().lower()
    text = strQJ2BJ(text)
    text = text.replace("（", "(") \
        .replace("[", "(") \
        .replace("{", "(") \
        .replace("【", "(") \
        .replace("「", "(") \
        .replace("）", ")") \
        .replace("）", ")") \
        .replace("]", ")") \
        .replace("}", ")") \
        .replace("】", ")") \
        .replace("」", ")")
    return text


if __name__ == '__main__':
    text = "ＴＣＬ集团股份有限公司技术（中心）"
    res = clean_norm(text)
    print(text)
    print(res)
    print("done")