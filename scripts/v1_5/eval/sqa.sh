#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_vqa_science \
    --model-path ./checkpoints/llava-v1.5-7b-lora-sq-2e4\
 	  --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/scienceqa/llava_test_CQM-A.json \
    --image-folder ./playground/data/eval/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-sq.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir ./playground/data/eval/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-sq.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-sq_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-sq_result.json
