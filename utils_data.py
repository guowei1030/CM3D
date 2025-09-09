#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/12/5 20:39
# @Author  : yebulk
import os
from torch.utils.data import Dataset
import os
import json
import numpy as np
import torch
from utils_prompt import *

img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (145, 1024),
}

def split_chinese_string(string):
    split_result = string.split("：", 1)
    if len(split_result) > 1:
        return split_result[1]
    else:
        return string

def load_data_std(args):
    problems = json.load(open(os.path.join(args.data_root, '../Data/output.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, '../Data/split.json')))
    # captions = json.load(open(args.caption_file))["captions"]

    # for qid in problems:
    #     problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val': val_qids, 'test': test_qids}
    return problems, qids,


def load_data_img(args):
    problems = json.load(open(os.path.join(args.data_root, 'output.json')))
    pid_splits = json.load(open(os.path.join(args.data_root, 'split.json')))
    # captions = json.load(open(args.caption_file))["captions"]

    name_maps = json.load(open('/mnt/sdb1/jsy/yjw/prompt/myCode/vision_features/name_map.json'))

    # check
    if args.img_type == "resnet":
        image_features = np.load('../vision_features_pics_Meme/resnet.npy')
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load('../vision_features_pics_Meme/clip.npy')
    elif args.img_type == "detr":
        image_features = torch.load('/mnt/sdb1/jsy/yjw/prompt/myCode/vision_features/detr.pth').to('cpu')
    elif args.img_type == "vit":
        image_features = torch.load("/mnt/sdb1/jsy/yjw/prompt/myCode/vision_features/vit.pth").to('cpu')
    else:
        image_features = np.load('/mnt/sdb1/jsy/yjw/prompt/myCode/vision_features/vit.pth')
    print("img_features size: ", image_features.shape)

    # for qid in problems:
    #     problems[qid]['caption'] = captions[qid] if qid in captions else ""

    train_qids = pid_splits['%s' % (args.train_split)]
    val_qids = pid_splits['%s' % (args.val_split)]
    test_qids = pid_splits['%s' % (args.test_split)]
    print(f"number of train problems: {len(train_qids)}\n")
    print(f"number of val problems: {len(val_qids)}\n")
    print(f"number of test problems: {len(test_qids)}\n")

    qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    return problems, qids, name_maps, image_features


class MetaphorDatasetStd(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
            self, problems, qids, tokenizer, answer_len, question_len, args, test_le=None
    ):
        self.tokenizer = tokenizer
        self.data = {qid: problems[qid] for qid in qids}
        self.answer_len = answer_len
        self.question_len = question_len
        self.questions = []
        self.answers = []
        if test_le is not None:
            test_le_data = json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            question, answer = build_train_pair(problems, qid, args, curr_le_data)
            self.questions.append(question)
            self.answers.append(answer)

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, index):
        question_text = str(self.questions[index])
        answer_text = str(self.answers[index])
        


        question = self.tokenizer.batch_encode_plus(
            [question_text],
            max_length=self.question_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        answer = self.tokenizer.batch_encode_plus(
            [answer_text],
            max_length=self.answer_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        question_ids = question["input_ids"].squeeze()
        question_mask = question["attention_mask"].squeeze()
        answer_ids = answer["input_ids"].squeeze().tolist()

        return {
            "input_ids": question_ids,
            "attention_mask": question_mask,
            "labels": answer_ids,
        }

class MeatphorDatasetImg(Dataset):
    """
        Creating a custom dataset for reading the dataset and
        loading it into the dataloader to pass it to the
        neural network for finetuning the model

        """

    def __init__(
            self, problems, qids, name_maps, tokenizer, question_len, answer_len, args, image_features, test_le=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = {qid: problems[qid] for qid in qids}
        self.answer_len = answer_len
        self.question_len = question_len
        self.questions = []
        self.answers = []
        self.image_ids = []
        if test_le is not None:
            test_le_data = json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            question, answer = build_train_pair(problems, qid, args, curr_le_data)
            self.questions.append(question)
            self.answers.append(answer)
            pic_name = str(problems[qid]['Pic_id'])
            if str(pic_name) in name_maps:
                i_vectors = image_features[int(name_maps[str(pic_name)])]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))

    def __len__(self):
        """returns the length of dataframe"""

        return len(self.answers)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        image_ids = self.image_ids[index]

        question_text = str(self.questions[index])
        answer_text = str(self.answers[index])

        question = self.tokenizer.batch_encode_plus(
            [question_text],
            max_length=self.question_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        answer = self.tokenizer.batch_encode_plus(
            [answer_text],
            max_length=self.answer_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        question_ids = question["input_ids"].squeeze()
        question_mask = question["attention_mask"].squeeze()
        answer_ids = answer["input_ids"].squeeze().tolist()

        image_ids = torch.tensor(image_ids).squeeze()

        return {
            "input_ids": question_ids,
            "attention_mask": question_mask,
            "labels": answer_ids,
            "image_ids": image_ids
        }
