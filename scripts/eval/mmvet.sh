#!/bin/bash

python -m llava.eval.model_vqa \
   --model-path ./checkpoints/path/to/ckpt\
	  --model-base lmsys/vicuna-7b-v1.5 \
   --question-file ./playground/data/eval/mm-vet/llava-mm-vet.jsonl \
   --image-folder ./playground/data/eval/mm-vet/images \
   --answers-file ./playground/data/eval/mm-vet/answers/vlorav2-665k-sq-clu.jsonl \
   --temperature 0 \
   --conv-mode vicuna_v1 \
   --lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora

mkdir -p ./playground/data/eval/mm-vet/results

python scripts/convert_mmvet_for_eval.py \
   --src ./playground/data/eval/mm-vet/answers/vlorav2-665k-sq-clu.jsonl \
   --dst ./playground/data/eval/mm-vet/results/vlorav2-665k-sq-clu.json


