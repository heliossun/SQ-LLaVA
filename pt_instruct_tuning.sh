PROMPT_VERSION="v1_sq"
MODEL_VERSION="lmsys/vicuna-7b-v1.5"

deepspeed --include localhost:1 train_mem.py \
    --deepspeed /home/gs4288/PycharmProjects/Visual-self-QA/scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path $MODEL_VERSION\
    --version $PROMPT_VERSION \
    --data_path /home/gs4288/data/NLP/llava_instruct_80k.json \
    --image_folder /home/gs4288/data/coco/images/train2017 \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type cross_attn \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/sqlva-$MODEL_VERSION\
    --num_train_epochs 0.01 \
    --per_device_train_batch_size 1 \
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
    --report_to wandb \
    --data_aug True