#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/path/to/ckpt \
 	--model-base ./checkpoints/sharegpt4_pretrain \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/llava-v1.5-13b.jsonl \
    --temperature 0 \
    --conv-mode v1 \
    --lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment llava-v1.5-13b

cd eval_tool

python calculation.py --results_dir answers/llava-v1.5-13b


