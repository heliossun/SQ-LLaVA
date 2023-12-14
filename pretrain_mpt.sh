#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=mtp1b-pret
#SBATCH --error=/home/gs4288/guohao/LLaVA/RC_error/err_%j.txt
#SBATCH --output=/home/gs4288/guohao/LLaVA/RC_out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=0-13:20:00
#SBATCH --gpus-per-node=a100:4
#SBATCH --partition tier3
#SBATCH --mem=100g
#SBATCH --account=crossmodal
#SBATCH --partition=tier3


source ~/conda/etc/profile.d/conda.sh
conda activate llava3

spack load gcc@9.3.0

deepspeed llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/mpt-1b-redpajama-200b-dolly \
    --version dolly \
    --data_path /home/gs4288/guohao/data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /home/gs4288/guohao/data/LLaVA-Pretrain \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/mtp1b-mlp-pretrain-1e3-4gpu \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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
    --data_aug False
