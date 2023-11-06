
###### LLaMA-2 lora eval
#flickr
# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-llama-2-7b-chat-lightning-lora-preview \
# 	--model_base ./checkpoints/llama-2-7b-chat \
# 	--question-file ./playground/flickr/question3.jsonl \
# 	--image-folder /home/gs4288/data/Flickr/flickr30k_images\
# 	--answers-file ./output/flickr/lora/answer_lm2_lora.jsonl \
# 	--conv-mode="llava_llama_2" \
# 	--pt \

# #coco
# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-llama-2-7b-chat-lightning-lora-preview \
# 	--model_base ./checkpoints/llama-2-7b-chat \
# 	--question-file ./playground/coco/question3.jsonl \
#  	--image-folder /home/gs4288/data/coco2014/images\
#  	--answers-file ./output/coco/lora/answer_lm2_lora.jsonl \
# 	--conv-mode="llava_llama_2" \
# 	--pt \

#concept	
# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-llama-2-7b-chat-lightning-lora-preview \
# 	--model_base ./checkpoints/llama-2-7b-chat \
# 	--question-file ./playground/conceptual/question3.jsonl \
# 	--image-folder /home/gs4288/data/conceptual \
# 	--answers-file ./output/conceptual/lora/answer_lm2_lora.jsonl \
# 	--conv-mode="llava_llama_2" \
# 	--pt

# #### vicuna-lora
# #concept	
# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-vicuna-v1-3-7b-finetune-lora \
# 	--model_base ./checkpoints/vicuna-v1-3-7b \
# 	--question-file ./playground/conceptual/question3.jsonl \
# 	--image-folder /home/gs4288/data/conceptual \
# 	--answers-file ./output/conceptual/lora/answer_vc_lora.jsonl \
# 	--conv-mode="v1" \
# 	--pt

# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-llava-llama-2-7b-chat-lora-pt\
# 	--model_base ./checkpoints/llava-llama-2-7b-chat-lora\
# 	--question-file ./playground/flickr/question3.jsonl \
# 	--image-folder /home/gs4288/data/Flickr/flickr30k_images\
# 	--answers-file ./output/flickr/lorapt/answer_lm2_pt_lora.jsonl \
# 	--conv-mode="llava_llama_2" \
# 	--pt \
# 	--ptlength 10

# # #coco
# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-llava-llama-2-7b-chat-lora-pt\
# 	--model_base ./checkpoints/llava-llama-2-7b-chat-lora\
# 	--question-file ./playground/coco/question3.jsonl \
#  	--image-folder /home/gs4288/data/coco2014/images\
#  	--answers-file ./output/coco/lorapt/answer_lm2_pt_lora.jsonl \
# 	--conv-mode="llava_llama_2" \
# 	--pt \
# 	--ptlength 10

# #concept	
# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-llava-llama-2-7b-chat-lora-pt\
# 	--model_base ./checkpoints/llava-llama-2-7b-chat-lora\
# 	--question-file ./playground/conceptual/question3.jsonl \
# 	--image-folder /home/gs4288/data/conceptual \
# 	--answers-file ./output/conceptual/lorapt/answer_lm2_pt_lora.jsonl \
# 	--conv-mode="llava_llama_2" \
# 	--pt \
# 	--ptlength 10

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/flickr/lorapt/answer_lm2_pt_lora.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--max_words 50\
# 	--model_name "llava_lm2_lora-pt"
# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/lorapt/answer_lm2_pt_lora.jsonl\
# 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
# 	--max_words 50\
# 	--model_name "llava_lm2_lora-pt"
# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/conceptual/lorapt/answer_lm2_pt_lora.jsonl\
# 	--annotation ./output/conceptual/ptlength50/answer_lm2_pt.jsonl \
# 	--max_words 50\
# 	--model_name "llava_lm2_lora-pt"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/flickr/answer_vicuna_lora_pt.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--max_words 50\
# 	--model_name "llava_vicuna_1_1"
#
# CUDA_VISIBLE_DEVICES=0 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-llava-v1.5-7b-pt-lr2e5\
# 	--model_base ./checkpoints/llava-v1.5-7b\
# 	--question-file ./playground/conceptual/question3.jsonl \
# 	--image-folder /home/gs4288/data/conceptual \
# 	--answers-file ./output/conceptual/lorapt/answer_vc_pt_1-5.jsonl \
# 	--conv-mode="v1" \
# 	--pt
#
# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/answer_lm2_pt_lora.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--max_words 50\
# 	--model_name "llava_vicuna_1_1"

# python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-llama-2-7b-chat-pt-finetune \
# 	--model_base ./checkpoints/llama-2-7b-chat \
# 	--question-file ./playground/conceptual/question3.jsonl \
# 	--image-folder /home/gs4288/data/conceptual \
# 	--answers-file ./output/conceptual/answer_lm2_pt_q3_new.jsonl \
# 	--conv-mode="llava_llama_2"

# python llava/eval/model_cap.py \
#  	--model_path ./checkpoints/llava-llama-2-7b-chat-pt-finetune \
# 	--model_base ./checkpoints/llama-2-7b-chat \
# 	--question-file ./playground/flickr/question3.jsonl \
# 	--image-folder /home/gs4288/data/Flickr/flickr30k_images\
# 	--answers-file ./output/flickr/answer_lm2_pt_q3_new.jsonl \
# 	--conv-mode="llava_llama_2"

# python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-llama-2-7b-chat-pt-finetune \
# 	--model_base ./checkpoints/llama-2-7b-chat \
# 	--question-file ./playground/coco/question3.jsonl \
# 	--image-folder /home/gs4288/data/coco2014/images\
# 	--answers-file ./output/coco/answer_lm2_pt_q3_new.jsonl \
# 	--conv-mode="llava_llama_2"


# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-llava-vicuna-7b-lora-pt\
# 	--model_base ./checkpoints/llava-vicuna-7b-lora \
# 	--question-file ./playground/flickr/question3.jsonl \
# 	--image-folder /home/gs4288/data/Flickr/flickr30k_images\
# 	--answers-file ./output/flickr/ptlength50/answer_vc_pt_lora_sq2.jsonl \
# 	--conv-mode="v1" \
# 	--pt \
# 	--ptlength 10

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/flickr/ptlength50/answer_vc_pt_lora_sq2.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--max_words 50\
# 	--model_name "llava_lm2_lora-pt"

#testmodel: llava-viduna 1.5
# #coco


#concept	
#CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_cap \
#  --model_path ./checkpoints/llava-v1.5-7b-lora\
#    --model_base ./checkpoints/llava-v1.5-7b\
#	--question-file ./playground/conceptual/question3.jsonl \
#	--image-folder /home/gs4288/data/conceptual \
#	--answers-file ./output/conceptual/answer_vc15.jsonl \
#	--conv-mode="v1" \
#	--temperature 0 \
#
#python llava/eval/eval_image_caption.py \
#	--answers-file ./output/conceptual/answer_vc15.jsonl\
#	--QA ./playground/conceptual/question3.jsonl \
#	--model_name "llava_vc15"\
#	--dataset "concept"

# CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_cap \
#       --model_path ./checkpoints/llava-v1.5-7b-lora-sq\
#  	--model_base ./checkpoints/llava-v1.5-7b\
#   	--question-file ./playground/flickr/question3.jsonl \
#   	--image-folder /home/gs4288/data/Flickr/flickr30k_images\
#   	--answers-file ./output/flickr/answer_llava1-5-lora-sq.jsonl \
#   	--conv-mode="v1_sq" \
#   	--sq \
#   	--temperature 0 \

#   python llava/eval/eval_image_caption.py \
#   	--answers-file ./output/flickr/answer_llava1-5-lora-sq.jsonl\
#   	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
#   	--model_name "llava_vc1-5-lora-sq" \
#   	--dataset "flickr"
 CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_cap \
    --model_path ./checkpoints/llava-v1.5-7b-lora-sq-2e4\
 	--model_base lmsys/vicuna-7b-v1.5\
 	--question-file ./playground/coco/question3.jsonl \
 	--image-folder /home/gs4288/data/coco2014/images\
 	--answers-file ./output/coco/answer_llava1-5.jsonl \
 	--conv-mode="v1_sq" \
 	--temperature 0 \
 	--sq

 python llava/eval/eval_image_caption.py \
 	--answers-file ./output/coco/answer_llava1-5.jsonl\
 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
 	--model_name "llava_vc1-5_lora_sq_os" \
 	--dataset "coco"

	CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_cap \
   --model_path ./checkpoints/llava-v1.5-7b-lora-sq-2e4\
    --model_base lmsys/vicuna-7b-v1.5\
   	--question-file ./playground/nocaps/near-domain_question.jsonl \
   	--image-folder /home/gs4288/data/nocaps/images/near-domain\
   	--answers-file ./output/nocaps/nd_answer_llava1-5.jsonl \
   	--conv-mode="v1_sq" \
  	--temperature 0 \
  	--sq

 python llava/eval/eval_image_caption.py \
 	--answers-file ./output/nocaps/nd_answer_llava1-5.jsonl\
 	--annotation /home/gs4288/data/nocaps/nocaps_val_near_domain.json\
 	--model_name "llava_vc1-5_lora_sq_os" \
 	--dataset "nocap_near_d"
#  CUDA_VISIBLE_DEVICES=0 python -m llava.eval.model_cap \
#     --model_path ./checkpoints/llava-v1.5-7b-lora-sq\
#  	--model_base ./checkpoints/llava-v1.5-7b\
#  	--question-file ./playground/coco/question3.jsonl \
#  	--image-folder /home/gs4288/data/coco2014/images\
#  	--answers-file ./output/coco/answer_llava1-5.jsonl \
#  	--conv-mode="v1" \
#  	--temperature 0 \
#
#  python llava/eval/eval_image_caption.py \
#  	--answers-file ./output/coco/answer_llava1-5.jsonl\
#  	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
#  	--model_name "llava_vc1-5_lora" \
#  	--dataset "coco"