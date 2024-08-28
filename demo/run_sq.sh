CUDA_VISIBLE_DEVICES=0 python visual_questioning.py \
  --model_path path/to/sqllava-v1.7-7b-lora-gpt4v-cluster-sq-vloraPTonly\
  --model_base Lin-Chen/ShareGPT4V-7B_Pretrained_vit-large336-l12_vicuna-7b-v1.5 \
  --conv-mode="v1_sq" \
  --lora_pretrain path/to/sqllava-v1.7-7b-lora-gpt4v-cluster-sq-vloraPTonly \
  --n_shot 3