PROMPT_VERSION="v1"
MODEL_VERSION="llava-v1.5-7b"

deepspeed --include localhost:1 llava/train/train_mem.py \
    --deepspeed /home/gs4288/LLaVA/scripts/zero2.json \
    --lora_enable True \
    --model_name_or_path /home/gs4288/LLaVA/checkpoints/$MODEL_VERSION\
    --version $PROMPT_VERSION \
    --data_path /home/gs4288/data/NLP/llava_instruct_aq_80k.json \
    --image_folder /home/gs4288/data/coco/images/train2017 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pt-10prompt-lr1e5\
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 12 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 5 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb 