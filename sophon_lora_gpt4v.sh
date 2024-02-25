deepspeed train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./mixTraindata/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
    --image_folder ./mixTraindata \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_lora  ./checkpoints/projector/Sophon-spt4v-v1.7-pretrain-ViT-LoRAv2-mlp/Vit-lora/adapter_model.bin \
    --pretrain_mm_mlp_adapter ./checkpoints/projector/Sophon-spt4v-v1.7-pretrain-ViT-LoRAv2-mlp/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-Sophon-v1.8-7b-vlorav2-shpt4v \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --vision_tower_lr 2e-5 \
    --vit_lora_enable \
    --lora_alpha_vit 64 \
    --lora_r_vit 32 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to None \
    --data_aug False \
    
    