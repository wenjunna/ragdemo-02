#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/6/12 11:18 PM
# @Author  : sunwenjun
# @File    : rag.py
# @brief: 大模型生成

import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # 读取本地 .env 文件，里面定义了 MOONSHOT_API_KEY和 MOONSHOT_API_URL

client = OpenAI(
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url=os.getenv("MOONSHOT_API_URL")
)


def get_completion(prompt, model="moonshot-v1-8k"):
    '''封装 openai 接口'''
    messages = [{
        "role": "system",
        "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"
                   "现在你的任务是根据下述给定的已知信息回答用户问题。确保你的回复完全依据下述已知信息。不要编造答案。如果下述已知信息不足以回答用户的问题，请直接回复'我无法回答您的问题'。"

    },
        {"role": "user", "content": prompt}]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,  # 模型输出的随机性，0 表示随机性最小
    )
    return response.choices[0].message.content


def build_prompt(prompt_template, **kwargs):
    '''
    根据Prompt模版生成Prompt
    :param prompt_template: 模版
    :param kwargs: 填充值
    :return:
    '''
    '''将 Prompt 模板赋值'''
    prompt = prompt_template
    for k, v in kwargs.items():
        if isinstance(v, str):
            val = v
        elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
            val = '\n'.join(v)
        else:
            val = str(v)
        prompt = prompt.replace(f"__{k.upper()}__", val)
    return prompt
