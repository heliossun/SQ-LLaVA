
 
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_cap \
    --model_path ./checkpoints/path/to/ckpt/\
    --model_base ./checkpoints/sharegpt4_pretrain \
 	--question-file ./playground/coco/question3.jsonl \
	--image-folder path/to/coco2014/images/val2014/\
 	--answers-file ./output/coco/1200.jsonl \
 	--conv-mode="v1" \
	--lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \


   python llava/eval/eval_image_caption.py \
   	--answers-file ./output/coco/1200.jsonl\
   	--annotation path/to/coco2014/annotations/coco_karpathy_test.json\
   	--model_name "1200" \
   	--dataset "coco"

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_cap \
    --model_path ./checkpoints/path/to/ckpt\
    --model_base ./checkpoints/sharegpt4_pretrain \
	--question-file ./playground/nocaps/near-domain_question.jsonl \
	--image-folder path/to/nocaps/images/near-domain\
	--answers-file ./output/nocaps/1200.jsonl \
	--conv-mode="v1" \
	--temperature 0 \
	--lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \
#
python llava/eval/eval_image_caption.py \
	--answers-file ./output/nocaps/1200.jsonl\
	--annotation path/to/nocaps/nocaps_val_near_domain.json\
	--model_name "1200" \
	--dataset "nocap_near_d"

CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_cap \
    --model_path ./checkpoints/path/to/ckpt \
    --model_base ./checkpoints/sharegpt4_pretrain\
	--question-file ./playground/flickr/question3.jsonl \
	--image-folder path/to/Flickr/flickr30k_images\
	--answers-file ./output/flickr/1200.jsonl \
	--conv-mode="v1" \
	--lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \

python llava/eval/eval_image_caption.py \
	--answers-file ./output/flickr/1200.jsonl\
	--annotation path/to/Flickr/annotations/flickr30k_test.json\
	--model_name "1200" \
	--dataset "flickr"

#
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_cap \
 --model_path ./checkpoints/path/to/ckpt/ \
  --model_base ./checkpoints/sharegpt4_pretrain\
 	--question-file ./playground/nocaps/out-domain_question.jsonl \
 	--image-folder /path/to/nocaps/images/out-domain\
 	--answers-file ./output/nocaps/1200.jsonl \
 	--conv-mode="v1" \
	--temperature 0 \
    --lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \

python llava/eval/eval_image_caption.py \
	--answers-file ./output/nocaps/1200.jsonl\
	--annotation path/to/nocaps/nocaps_val_out_domain.json\
	--model_name "1200" \
	--dataset "nocap_out_d"
#
CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_cap \
 --model_path ./checkpoints/path/to/ckpt \
   --model_base ./checkpoints/sharegpt4_pretrain\
	--question-file ./playground/conceptual/question3.jsonl \
	--image-folder /path/to/conceptual \
	--answers-file ./output/conceptual/1200.jsonl \
	--conv-mode="v1" \
	--temperature 0 \
	--lora_pretrain ./checkpoints/path/to/ckpt/Vit-lora \

python llava/eval/eval_image_caption.py \
	--answers-file ./output/conceptual/1200.jsonl\
	--QA ./playground/conceptual/question3.jsonl \
	--model_name "1200"\
	--dataset "concept"
