# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/answer_llava_vicuna_1_1.jsonl\
# 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
# 	--max_words 20\
# 	--model_name "llava_vicuna_1_1"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/answer_llava_vicuna_1_1.jsonl\
# 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
# 	--max_words 50\
# 	--model_name "llava_vicuna_1_1"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/answer_llava_vicuna_1_1.jsonl\
# 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
# 	--max_words 100\
# 	--model_name "llava_vicuna_1_1"



#python llava/eval/eval_image_caption.py \
#	--answers-file ./output/coco/answer_llava_vicuna_lora_q2.jsonl\
#	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
#	--max_words 20\
#	--model_name "llava_vicuna_lora"
# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/answer_vicuna_lora_q3.jsonl\
# 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
# 	--max_words 50\
# 	--model_name "llava_vicuna_lora"
#python llava/eval/eval_image_caption.py \
#	--answers-file ./output/coco/answer_llava_vicuna_lora_q2.jsonl\
#	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
#	--max_words 100\
#	--model_name "llava_vicuna_lora"
#
#
# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/vinvl_result.json\
# 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
# 	--max_words 20\
# 	--model_name "llava_lm2_lora"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/flickr/answer_llava_pt_lora_q3.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--max_words 50\
# 	--model_name "llava_lm2_lora-pt"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/flickr/answer_vicuna_lora_q3.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--max_words 50\
# 	--model_name "llava_vicuna_lora"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/conceptual/answer_vicuna1-1_q3.jsonl \
# 	--QA ./playground/conceptual/question3.jsonl \
# 	--max_words 50\
# 	--model_name "llava_vicuna_1_1"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/conceptual/answer_lm2_pt_q3_new.jsonl \
# 	--QA ./playground/conceptual/question3.jsonl \
# 	--max_words 50\
# 	--model_name "llava_vicuna_1_1"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/conceptual/answer_vicuna_pt_q3.jsonl \
# 	--QA ./playground/conceptual/question3.jsonl \
# 	--max_words 50\
# 	--model_name "llava_vicuna_1_1"

#  python llava/eval/eval_image_caption.py \
#  	--answers-file ./output/coco/answer_clip_q3_new.jsonl \
#  	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
#  	--max_words 50\
#  	--model_name "llava_vicuna_1_1"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/flickr/answer_lm2_pt_q3_new.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--max_words 50\
# 	--model_name "llava_vicuna_1_1"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/answer_vicuna_mmpt_q3.jsonl\
# 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
# 	--max_words 50\
# 	--model_name "llava_vicuna_1_1"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/flickr/answer_vicuna_mmptv2_q3.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--max_words 50\
# 	--model_name "llava_vicuna_1_1"

# python llava/eval/model_cap.py \
#     --model_path ./checkpoints/LLaVA-7B-v1-1 \
# 	--question-file ./playground/conceptual/question3.jsonl \
# 	--image-folder /home/gs4288/data/conceptual \
# 	--answers-file ./output/conceptual/answer_vicuna1-1_q3.jsonl \
# 	--conv-mode="v1"


###### LLaMA-2 pt eval
#flickr
# CUDA_VISIBLE_DEVICES=1 python -m llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-v1.5-7b-lora-sq2-665k\
# 	--model_base lmsys/vicuna-7b-v1.5 \
# 	--question-file ./playground/flickr/question3.jsonl \
# 	--image-folder /home/gs4288/data/Flickr/flickr30k_images\
# 	--answers-file ./output/flickr/answer_llava1-5-lora.jsonl \
# 	--conv-mode="v1" \
#  --temperature 0 \

#  python llava/eval/eval_image_caption.py \
#  	--answers-file ./output/flickr/answer_llava1-5-lora.jsonl\
#  	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
#  	--model_name "llava_vc1-5-lora" \
#  	--dataset "flickr"

# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-v1.5-7b\
# 	--question-file ./playground/flickr/question3.jsonl \
# 	--image-folder /home/gs4288/data/Flickr/flickr30k_images\
# 	--answers-file ./output/flickr/answer_llava1-5.jsonl \
# 	--conv-mode="v1" \
# 	--temperature 0 \
#
# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/flickr/answer_llava1-5.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--model_name "llava_vc1-5" \
# 	--dataset "flickr"

# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#    --model_path ./checkpoints/llava-v1.5-7b-lora-336\
# 	--model_base ./checkpoints/llava-v1.5-7b\
# 	--question-file ./playground/coco/question3.jsonl \
# 	--image-folder /home/gs4288/data/coco2014/images\
# 	--answers-file ./output/coco/answer_llava1-5.jsonl \
# 	--conv-mode="v1" \
# 	--temperature 0 \

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/answer_llava1-5.jsonl\
# 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
# 	--model_name "llava_vc1-5_lora" \
# 	--dataset "coco"

	#nocaps

# CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_cap \
#    --model_path ./checkpoints/llava-v1.5-7b-lora-sq2-665k\
#     --model_base lmsys/vicuna-7b-v1.5 \
#    	--question-file ./playground/nocaps/out-domain_question.jsonl \
#    	--image-folder /home/gs4288/data/nocaps/images/out-domain\
#    	--answers-file ./output/nocaps/od_answer_llava1-5-lora665k-sq2.jsonl \
#    	--conv-mode="v1_sq" \
#   	--temperature 0 \
#   	--sq

#  python llava/eval/eval_image_caption.py \
#  	--answers-file ./output/nocaps/od_answer_llava1-5-lora665k-sq2.jsonl\
#  	--annotation /home/gs4288/data/nocaps/nocaps_val_out_domain.json\
#  	--model_name "llava_vc1-5_lora-sq" \
#  	--dataset "nocap_out_d"

# CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_cap \
#    --model_path ./checkpoints/llava-v1.5-7b-lora-sq-2e4-10epo\
#     --model_base lmsys/vicuna-7b-v1.5 \
#    	--question-file ./playground/nocaps/out-domain_question.jsonl \
#    	--image-folder /home/gs4288/data/nocaps/images/out-domain\
#    	--answers-file ./output/nocaps/od_answer_llava1-5-lora.jsonl \
#    	--conv-mode="v1_sq" \
#   	--temperature 0 \
#   	--sq

#  python llava/eval/eval_image_caption.py \
#  	--answers-file ./output/nocaps/od_answer_llava1-5-lora.jsonl\
#  	--annotation /home/gs4288/data/nocaps/nocaps_val_out_domain.json\
#  	--model_name "llava_vc1-5_lora-sq" \
#  	--dataset "nocap_out_d"
#
#
CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_cap \
 --model_path ./checkpoints/llava-v1.5-7b-lora-sq2-665k\
	--model_base lmsys/vicuna-7b-v1.5 \
	--question-file ./playground/conceptual/question3.jsonl \
	--image-folder /home/gs4288/data/conceptual \
	--answers-file ./output/conceptual/answer_vc15_lora.jsonl \
	--conv-mode="v1" \
	--temperature 0 \

python llava/eval/eval_image_caption.py \
	--answers-file ./output/conceptual/answer_vc15_lora.jsonl\
	--QA ./playground/conceptual/question3.jsonl \
	--model_name "llava_vc15_lora"\
	--dataset "concept"
#
#  CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_cap \
#      --model_path ./checkpoints/llava-v1.5-7b-lora-sq2-2e4\
# 	--model_base ./checkpoints/vicuna-7b-v1.5\
#  	--question-file ./playground/flickr/question3.jsonl \
#  	--image-folder /home/gs4288/data/Flickr/flickr30k_images\
#  	--answers-file ./output/flickr/answer_llava1-5-lora.jsonl \
#  	--conv-mode="v1" \
#  	--temperature 0 \

#  python llava/eval/eval_image_caption.py \
#  	--answers-file ./output/flickr/answer_llava1-5-lora.jsonl\
#  	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
#  	--model_name "llava_vc1-5-lora" \
#  	--dataset "flickr"

#  CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_cap \
#     --model_path ./checkpoints/llava-v1.5-7b-lora-sq\
#  	--model_base ./checkpoints/llava-v1.5-7b\
#  	--question-file ./playground/coco/question3.jsonl \
#  	--image-folder /home/gs4288/data/coco2014/images\
#  	--answers-file ./output/coco/answer_llava1-5.jsonl \
#  	--conv-mode="v1_sq" \
#  	--temperature 0 \
#
#  python llava/eval/eval_image_caption.py \
#  	--answers-file ./output/coco/answer_llava1-5.jsonl\
#  	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
#  	--model_name "llava_vc1-5_lora_sq" \
#  	--dataset "coco"

# CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#    --model_path ./checkpoints/llava-llava-v1.5-7b-pt-lr2e5\
# 	--model_base ./checkpoints/llava-v1.5-7b\
# 	--question-file ./playground/flickr/question3.jsonl \
# 	--image-folder /home/gs4288/data/Flickr/flickr30k_images\
# 	--answers-file ./output/flickr/answer_llava1-5_pt.jsonl \
# 	--conv-mode="v1" \
# 	--pt \
# 	--do_sample \
# 	--ptlength 10

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/flickr/answer_llava1-5_pt.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--model_name "llava_vc1-5_pt" \
# 	--dataset "flickr"

#CUDA_VISIBLE_DEVICES=1 python -m llava.eval.model_cap \
#   --model_path ./checkpoints/llava-v1.5-7b-lora\
#	--model_base ./checkpoints/llava-v1.5-7b\
#	--question-file ./playground/coco/question3.jsonl \
#	--image-folder /home/gs4288/data/coco2014/images\
#	--answers-file ./output/coco/answer_llava1-5_pt.jsonl \
#	--conv-mode="v1_sq" \
#	--pt \
#	--sq \
#	--temperature 0 \
#	--ptlength 10
#
#python llava/eval/eval_image_caption.py \
#	--answers-file ./output/coco/answer_llava1-5_pt.jsonl\
#	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
#	--model_name "llava_vc1-5_pt" \
#	--dataset "coco"
#    CUDA_VISIBLE_DEVICES=1 python llava/eval/model_cap.py \
#    --model_path ./checkpoints/llava-v1.5-7b-lora-336\
#     --model_base ./checkpoints/llava-v1.5-7b\
#    	--question-file ./playground/nocaps/out-domain_question.jsonl \
#    	--image-folder /home/gs4288/data/nocaps/images/out-domain\
#    	--answers-file ./output/nocaps/od_answer_llava1-5.jsonl \
#    	--conv-mode="v1" \
#   	--temperature 0 \

#  python llava/eval/eval_image_caption.py \
#  	--answers-file ./output/nocaps/od_answer_llava1-5.jsonl\
#  	--annotation /home/gs4288/data/nocaps/nocaps_val_out_domain.json\
#  	--model_name "llava_vc1-5-pt" \
#  	--dataset "nocap_out_d"

# #coco
# CUDA_VISIBLE_DEVICES=0 python llava/eval/model_cap.py \
#     --model_path ./checkpoints/llava-vicuna-7b-lora-pt-sq\
# 	--model_base ./checkpoints/llava-vicuna-7b-lora \
# 	--question-file ./playground/coco/question3.jsonl \
#  	--image-folder /home/gs4288/data/coco2014/images\
#  	--answers-file ./output/coco/sq/answer_vc.jsonl \
# 	--conv-mode="v1_sq" \
# 	--pt \
# 	--sq \
# 	--ptlength 10


# #concept	


#TODO


# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/sq/answer_vc.jsonl\
# 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
# 	--max_words 50\
# 	--model_name "llava-vicuna-7b-lora-pt-sq" \
# 	--dataset "coco"

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
# 	--QA ./playground/conceptual/question3.jsonl \
# 	--max_words 50\
# 	--model_name "llava_lm2_lora-pt"



# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/coco/ptlength50/answer_lm2_pt.jsonl\
# 	--annotation /home/gs4288/data/coco2014/annotations/coco_karpathy_test.json\
# 	--max_words 50\
# 	--model_name "llava_lm2_lora"

# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/flickr/ptlength50/answer_lm2_pt.jsonl\
# 	--annotation /home/gs4288/data/Flickr/annotations/flickr30k_test.json\
# 	--max_words 50\
# 	--model_name "llava_lm2_lora-pt"


# python llava/eval/eval_image_caption.py \
# 	--answers-file ./output/conceptual/ptlength50/answer_lm2_pt.jsonl \
# 	--QA ./playground/conceptual/question3.jsonl \
# 	--max_words 50\
# 	--model_name "llava_vicuna_1_1"

