#!/bin/bash

SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path ./checkpoints/path/to/ckpt/ \
 	  --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/vlora-cluster-sq30.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode v1 \
    --lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment vlora-cluster-sq30



