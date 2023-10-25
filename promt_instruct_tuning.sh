#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=llava
#SBATCH --error=/home/gs4288/guohao/LLaVA/RC_error/err_%j.txt
#SBATCH --output=/home/gs4288/guohao/LLaVA/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=00-0:10:00
#SBATCH --gpus-per-node=a100:1
#SBATCH --partition tier3
#SBATCH --mem=64g
#SBATCH --account=crossmodal
#SBATCH --partition=tier3


source ~/conda/etc/profile.d/conda.sh
conda activate llava2

PROMPT_VERSION="llava_llama_2"
MODEL_VERSION="llama-2-7b-chat"

deepspeed llava/train/train_mem.py \
    --deepspeed /home/gs4288/guohao/LLaVA/scripts/zero2.json \
    --model_name_or_path /home/gs4288/guohao/LLaVA/checkpoints/$MODEL_VERSION\
    --version $PROMPT_VERSION \
    --data_path /home/gs4288/guohao/data/NLP/llava_instruct_80k.json \
    --image_folder /home/gs4288/guohao/data/coco/images/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --pretrain_mm_mlp_adapter /home/gs4288/guohao/LLaVA/checkpoints/llava-$MODEL_VERSION-pretrain/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-finetune \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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