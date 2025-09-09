'''
Adapted from https://github.com/lupantech/ScienceQA
'''

import os
import json
import argparse
import warnings
import pandas as pd
from sentence_transformers import SentenceTransformer
from bert_score import score
from evaluations import caculate_bleu, caculate_rouge, caculate_similariry
from utils_data import split_chinese_string

warnings.filterwarnings('ignore')

def get_acc_with_contion(res_pd, key, values):
    if isinstance(values, list):
        total_pd = res_pd[res_pd[key].isin(values)]
    else:
        total_pd = res_pd[res_pd[key] == values]
    correct_pd = total_pd[total_pd['true_false'] == True]
    acc = "{:.2f}".format(len(correct_pd) / len(total_pd) * 100)
    return acc

def calculate_accuracy(dict1, dict2):
    common_keys = set(dict1.keys()) & set(dict2.keys())
    total_keys = len(common_keys)
    equal_count = 0

    for key in common_keys:
        if dict1[key] == dict2[key]:
            equal_count += 1

    accuracy = equal_count / total_keys if total_keys > 0 else 0
    return accuracy

def process_dict_values(dictionary):
    processed_dict = {}
    for key, value in dictionary.items():
        processed_dict[key] = split_chinese_string(value)
    return processed_dict

def get_scores(result_data, rationale_data, results_reference, data_file):
    # read result file
    results = result_data
    num = len(results)
    rationale_data = process_dict_values(rationale_data)
    results_reference = process_dict_values(results_reference)
    rationale_data_list = list(rationale_data.values())
    results_reference_list = list(results_reference.values())
    ## BLEU
    bleu1 = caculate_bleu(rationale_data, results_reference, gram=1)
    bleu4 = caculate_bleu(rationale_data, results_reference, gram=4)


    correct = calculate_accuracy(rationale_data, results_reference)

    ## Rouge-L
    # rouge = caculate_rouge(rationale_data, results_reference)
    P, R, F1 = score(rationale_data_list, results_reference_list, model_type="bert-base-chinese", lang="zh", verbose=True)
    ## Similarity
    model = SentenceTransformer('/mnt/sdb1/jsy/yjw/prompt/my-cot-metaphor/modals/all-MiniLM-L6-v2').cuda()
    similariry = caculate_similariry(rationale_data, results_reference, model)

    scores = {

            "rationale":{
                'bleu1': bleu1 * 100,
                'bleu4': bleu4 * 100,
                'BertAcore': F1.mean().item()*100,
                'similariry': similariry * 100,
                'acc': correct * 100
            }
    }

    return scores



