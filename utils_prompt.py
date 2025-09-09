#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/12/5 22:13
# @Author  : yebulk

def get_target_text(data):
    target = data['Target']
    return target

def get_source_text(data):
    target = data['Source']
    return target

def get_text(data):
    target = data['Text']
    return target

def get_propose(data):
    propose = data['Propose']
    return propose

def build_train_pair(data, test_qid, args, curr_le_data=None):
    examples = []

    # test example
    target = get_target_text(data[test_qid])
    source = get_source_text(data[test_qid])
    text = get_text(data[test_qid])
    propose = ""
    exception = ""
    type = ""
    # 获取目的
    # propose = get_propose_text(data[test_qid])


    question, answer = create_one_example(args.prompt_format, target, source, text, propose, exception, type,  WithOutput = True, test_example=False)

    examples.append(question)
    return question, answer

def build_pair(data, test_qid, args, curr_le_data=None):
    examples = []

    # test example
    target = get_target_text(data[test_qid])
    source = get_source_text(data[test_qid])
    text = get_text(data[test_qid])
    propose = ""
    exception = ""
    type = ""
    # 获取目的
    # propose = get_propose_text(data[test_qid])

    question, answer = create_one_example(args.prompt_format, target, source, text, propose, exception, type,
                                          WithOutput=True, test_example=False)

    examples.append(question)

    return question, answer

def create_one_example(prompt_format, target, source, text, propose, exception, type, WithOutput = False, test_example=True):
    if len(propose) > 30:
        propose = ""
    if prompt_format == "Stage_One":
        if test_example:
            # question = f"图片的文本是{text}，这是一张{type}图片， 目的是{propose}, 目标域："
            question = f"图片的文本是{text}，这是一张{type}图片， 目的是{propose}, 目标域："
            print(question)
            answer = f"目标域:"

        else:
            # question = f"这是一张{type}图片， 目的是{propose}, 目标域是什么"
            # question = f"图片的文本是{text}，这是一张{type}图片， 目的是{propose}, 目标域："
            question = f"目标域?"
            answer = f"{target}"
    else:
        if test_example:
            question = f"图片的文本是{text}，{exception}，源域："
            answer = f"源域："
        else:
            # question = f"图片的文本是{text}，{exception}，源域："
            question = f"源域?"
            answer = f"{source}"
    return question, answer
