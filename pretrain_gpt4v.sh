#!/bin/bash
deepspeed train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/ShareGPT4V-13B_Pretrained_vit-large336-l12_vicuna-13b-v1.5 \
    --version plain \
    --data_path ./mixTraindata/share-captioner_coco_lcs_sam_1246k_1107.json \
    --image_folder ./mixTraindata\
    --vision_tower Lin-Chen/ShareGPT4V-13B_Pretrained_vit-large336-l12 \
    --pretrain_mm_mlp_adapter ./checkpoints/projector/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/projector/sqllava-spt4v-v1.7-13b-pretrain-mlp \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --data_aug False


# MLP pretrain 13b: sqllava-spt4v-v1.7-13b-pretrain-mlp
# MLP pretrain 7b: sqllava-spt4v-v1.7-pretrain-mlp