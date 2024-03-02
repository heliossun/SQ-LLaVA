deepspeed train_mem.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/ShareGPT4V-13B_Pretrained_vit-large336-l12_vicuna-13b-v1.5 \
    --version v1_sq \
    --data_path ./mixTraindata/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
    --image_folder ./mixTraindata \
    --vision_tower Lin-Chen/ShareGPT4V-13B_Pretrained_vit-large336-l12 \
    --pretrain_mm_mlp_adapter ./checkpoints/projector/sqllava-shpt4-13b-spt4v-v1.7-pretrain-cluster/mm_projector.bin \
    --mm_projector_type cluster \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/sqllava-v1.7-13b-lora-gpt4v-vloraPTonly-cluster-sq50 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --vision_tower_lr 2e-4 \
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
    --sq_r 0.5\
    
    