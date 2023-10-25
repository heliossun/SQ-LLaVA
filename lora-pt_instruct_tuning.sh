#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=llava-qa
#SBATCH --error=/home/gs4288/guohao/LLaVA/RC_error/err_%j.txt
#SBATCH --output=/home/gs4288/guohao/LLaVA/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=1-10:10:00
#SBATCH --gpus-per-node=a100:2
#SBATCH --partition tier3
#SBATCH --mem=100g
#SBATCH --account=crossmodal
#SBATCH --partition=tier3


source ~/conda/etc/profile.d/conda.sh
conda activate llava3

spack load gcc@9.3.0
PROMPT_VERSION="v1"
MODEL_VERSION="vicuna-v1-3-7b"

deepspeed llava/train/train_mem.py \
    --deepspeed /home/gs4288/guohao/LLaVA/scripts/zero2.json \
    --pt_enable True \
    --pt_method "pt" \
    --encoder_type 'MLP' \
    --model_name_or_path /home/gs4288/guohao/LLaVA/checkpoints/$MODEL_VERSION\
    --version $PROMPT_VERSION \
    --data_path /home/gs4288/guohao/data/NLP/llava_instruct_aq_80k.json \
    --image_folder /home/gs4288/guohao/data/coco/images/train2017 \
    --vision_tower openai/clip-vit-large-patch14 \
    --tune_mm_mlp_adapter \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-$MODEL_VERSION-pt-10prompt-lr1e5\
    --num_train_epochs 5 \
    --per_device_train_batch_size 25 \
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
    