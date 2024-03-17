#!/bin/bash

if [ "$1" = "dev" ]; then
    echo "Evaluating in 'dev' split."
elif [ "$1" = "test" ]; then
    echo "Evaluating in 'test' split."
else
    echo "Unknown split, please choose between 'dev' and 'test'."
    exit 1
fi

 python -m llava.eval.model_vqa_qbench \
    --model-path ./checkpoints/path/to/ckpt \
 	  --model-base ./checkpoints/sharegpt4_pretrain  \
    --image-folder ./playground/data/eval/qbench/images_llvisionqa/ \
    --questions-file ./playground/data/eval/qbench/llvisionqa_$1.json \
    --answers-file ./playground/data/eval/qbench/llvisionqa_$1_answers.jsonl \
    --conv-mode llava_v1 \
    --lang en \
    --lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \

python playground/data/eval/qbench/format_qbench.py \
    --filepath ./playground/data/eval/qbench/llvisionqa_$1_answers.jsonl

python playground/data/eval/qbench/qbench_eval.py \
    --filepath ./playground/data/eval/qbench/llvisionqa_$1_answers.jsonl