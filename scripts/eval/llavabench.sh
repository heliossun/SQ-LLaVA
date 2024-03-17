#!/bin/bash


python -m llava.eval.model_vqa \
    --model-path ./checkpoints/path/to/ckpt\
 	  --model-base lmsys/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
    --image-folder ./playground/data/eval/llava-bench-in-the-wild/images \
    --answers-file ./playground/data/eval/llava-bench-in-the-wild/answers/vloraPTonly-665k-sq50-clu.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora

mkdir -p playground/data/eval/llava-bench-in-the-wild/reviews

python llava/eval/eval_gpt_review_bench.py \
   --question playground/data/eval/llava-bench-in-the-wild/questions.jsonl \
   --context playground/data/eval/llava-bench-in-the-wild/context.jsonl \
   --rule llava/eval/table/rule.json \
   --answer-list \
       playground/data/eval/llava-bench-in-the-wild/answers_gpt4.jsonl \
       playground/data/eval/llava-bench-in-the-wild/answers/13bvloraPTonly-665k-sq50-clu.jsonl \
   --output \
       playground/data/eval/llava-bench-in-the-wild/reviews/13bvloraPTonly-665k-sq50-clu.jsonl

python llava/eval/summarize_gpt_review.py -f playground/data/eval/llava-bench-in-the-wild/reviews/13bvloraPTonly-665k-sq50-clu.jsonl




