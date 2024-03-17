#!/bin/bash

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/path/to/ckpt \
 	--model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-gpt4v.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/llava-v1.5-7b-gpt4v.jsonl



