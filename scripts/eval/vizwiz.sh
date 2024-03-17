#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/path/to/ckpt\
    --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/665k-vlora-cluster.jsonl \
    --temperature 0 \
    --conv-mode v1 \
    --lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \

python -m scripts.convert_vizwiz_for_submission \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/665k-vlora-cluster.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/665k-vlora-cluster.json


