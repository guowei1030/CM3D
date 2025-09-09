#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time    : 2023/12/5 20:34
# @Author  : yebulk

import os
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import re
import json
import argparse
import random
from transformers import BertTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from BartModel import BartForMultimodalGeneration
from utils_data import img_shape, load_data_std, load_data_img, MetaphorDatasetStd, MeatphorDatasetImg, split_chinese_string
from utils_prompt import *
from utils_evaluate import get_scores
from rich.table import Column, Table
from rich import box
from rich.console import Console
from bert_score import score

console = Console(record=True)
import nltk

nltk.download('punkt')
import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../Data/')
    parser.add_argument('--output_dir', type=str, default='experiments—Ours-notexts')
    parser.add_argument('--model', type=str, default='../modals/bart-large-chinese')
    # parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=16)
    parser.add_argument('--output_len', type=int, default=16)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'debug_train_set', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])

    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="source", choices=['target', 'source'], help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default='vit', choices=['detr', 'clip', 'resnet', 'vit'],
                        help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/instruct_captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--prompt_format', type=str, default="Stage_two", help='prompt format template',
                        choices=['Stage_One', 'Stage_two'])
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args


def BARTTrainer(
        dataframe, args,
):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = BertTokenizer.from_pretrained(args.model)

    console.log(f"""[Model]:  {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    problems = dataframe['problems']
    qids = dataframe['qids']
    train_qids = qids['train']
    test_qids = qids['test']
    val_qids = qids['val']

    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/", "-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    print(save_dir)

    if args.img_type is not None:
        patch_size = img_shape[args.img_type]
        model = BartForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size)
        name_maps = dataframe['name_maps']
        image_features = dataframe['image_features']
        train_set = MeatphorDatasetImg(
            problems,
            train_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
        ) # 4886
        eval_set = MeatphorDatasetImg(
            problems,
            val_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.eval_le,
        ) # 610
        test_set = MeatphorDatasetImg(
            problems,
            test_qids,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.test_le,
        )# 612
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
        train_set = MetaphorDatasetStd(
            problems,
            train_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
        )
        eval_set = MetaphorDatasetStd(
            problems,
            val_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.eval_le,
        )

        test_set = MetaphorDatasetStd(
            problems,
            test_qids,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            args.test_le,
        )

    datacollator = DataCollatorForSeq2Seq(tokenizer)
    print("model parameters: ", model.num_parameters())

#
#         # accuracy for answer inference
#
#     def compute_metrics_acc(eval_preds):
#         if args.use_generate:
#             preds, targets = eval_preds
#             if isinstance(preds, tuple):
#                 preds = preds[0]
#         else:
#             preds = eval_preds.predictions[0]
#             targets = eval_preds.label_ids
#             preds = preds.argmax(axis=2)
#         preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#         correct = 0
#         assert len(preds) == len(targets)
#         for idx, pred in enumerate(preds):
#             reference = targets[idx]
#             reference = extract_ans(reference)
#             extract_pred = extract_ans(pred)
#             best_option = extract_pred
#             if reference == best_option:
#                 correct += 1
#         return {'accuracy': 1.0 * correct / len(targets)}
#
#     # rougel for rationale generation
    metric = evaluate.load("/mnt/sdb1/jsy/yjw/prompt/my-cot-metaphor/metrics/rouge")
#
#
    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            # print("preds", preds)
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds = split_chinese_string(preds)
        decoded_labels = split_chinese_string(targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    def compute_metrics_Bertscore(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)

        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        cands = preds
        refs = targets

        P, R, F1 = score(cands, refs, model_type="bert-base-chinese", lang="zh", verbose=True)
        print(F1)
        print(f"System level F1 score: {F1.mean():.3f}")

        # 构建包含P、R和F1的字典
        result_dict = {
            "P": P.mean().item(),
            "R": R.mean().item(),
            "F1": F1.mean().item()
        }

        return result_dict

# #
#     # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            report_to="none",
        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            save_total_limit=2,
            learning_rate=args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="F1",
            # metric_for_best_model="accuracy" if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else "rougeL",
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            load_best_model_at_end=True,
            report_to="none",
        )
# #
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics_Bertscore
        # compute_metrics=compute_metrics_rougel
        # compute_metrics=compute_metrics_acc if args.prompt_format == "QCMG-A" or args.prompt_format == "QCM-A" else compute_metrics_rougel
    )
# #
    if args.evaluate_dir is None:
        trainer.train()
        trainer.save_model(save_dir)
    print("save_dir", save_dir)
    metrics = trainer.evaluate(eval_dataset=test_set, max_length=args.output_len)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
# #
    predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len)
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids
            preds = preds.argmax(axis=2)

        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )

        results_ans = {}
        results_rationale = {}
        results_reference = {}

        num_fail = 0
        for idx, qid in enumerate(test_qids):
            pred = preds[int(idx)]
            ref = targets[int(idx)]
            results_rationale[str(qid)] = pred
            results_reference[str(qid)] = ref

        scores = get_scores(results_ans, results_rationale, results_reference,
                            os.path.join(args.data_root, "scienceqa/problems.json"))
        preds = [pred.strip() for pred in preds]
        output_data = {
            "num_fail": num_fail,
            "scores": scores,
            "preds": preds,
            "labels": targets}
        output_prediction_file = os.path.join(save_dir, "predictions_ans_test.json")
        with open(output_prediction_file, "w" ,encoding="utf-8") as writer:
            writer.write(json.dumps(output_data, indent=4, ensure_ascii=False))

    # generate the rationale for the eval set
    # if args.prompt_format == "QCM-LE" or args.prompt_format == "QCM-E":
    torch.cuda.empty_cache()
    del predict_results, preds, targets
    predict_results = trainer.predict(test_dataset=eval_set, max_length=args.output_len)
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids
            preds = preds.argmax(axis=2)

        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        preds = [pred.strip() for pred in preds]
        output_data = {"preds": preds,
                       "labels": targets}
        output_prediction_file = os.path.join(save_dir, "predictions_ans_eval.json")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            writer.write(json.dumps(output_data, indent=4, ensure_ascii=False))


if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )

    args = parse_args()
    print("args", args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.img_type is not None:
        problems, qids, name_maps, image_features = load_data_img(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems': problems, 'qids': qids, 'name_maps': name_maps, 'image_features': image_features}
    else:
        problems, qids = load_data_std(args)  # probelms, test question ids, shot example ids
        dataframe = {'problems': problems, 'qids': qids}

    BARTTrainer(
        dataframe=dataframe,
        args=args
    )
