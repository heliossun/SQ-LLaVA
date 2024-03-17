#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/path/to/ckpt/ \
 	  --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-lora-665k-cq.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode v1 \
    --lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \


python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-lora-665k-cq.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-lora-665k-cq_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-lora-665k-cq_result.json

