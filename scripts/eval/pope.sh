

python -m llava.eval.model_vqa_loader \
    --model-path ./checkpoints/path/to/ckpt\
 	  --model-base ./checkpoints/sharegpt4_pretrain \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder /path/to/coco2014/images/val2014 \
    --answers-file ./playground/data/eval/pope/answers/llava-v1.5-7b-gpt4v.jsonl \
    --temperature 0 \
    --conv-mode "v1" \
    --lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \

python -m llava.eval.eval_pope \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/llava-v1.5-7b-gpt4v.jsonl

