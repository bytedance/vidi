#!/bin/bash
DATE="$(date +%m%d)"
OUTPUT_DIR="checkpoint/YOUR_DIR"
MM_RAND_LR=2e-5
LR=1e-5
SP=1
MM_SPLITS=4
BS=1
WORKER_NUM=1
let GA=16*$SP/$BS/$WORKER_NUM
 
export DECORD_EOF_RETRY_MAX=20480 

deepspeed --master_port ${YourPort} vidi/train/train.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path "Your Model Path" \
    --llm_attn "dattn" \
    --mm_vision_tower "google/siglip2-so400m-patch14-384" \
    --mm_vision_select_layer -2 \
    --mm_image_aspect_ratio "resize" \
    --mm_image_pool_size 2 \
    --mm_audio_tower "openai/whisper-large-v3" \
    --mm_audio_pool_size 5 \
    --mm_input_type "video" \
    --mm_std 0.028976401314139366 \
    --dataset_type "video-conv" \
    --loss_thres 0.1 \
    --video_folder "." \
    --data_path "example.json" \
    --bf16 true \
    --tf32 true \
    --seq_parallel_size $SP \
    --mm_splits $MM_SPLITS \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BS \
    --gradient_accumulation_steps $GA \
    --eval_strateg "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 2 \
    --train_rand true \
    --train_vis false \
    --train_aud false \
    --train_llm true \
    --mm_rand_lr $MM_RAND_LR \
    --learning_rate $LR \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-5 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_strategy "steps" \
    --logging_steps 1 \
    --dataloader_num_workers 4 \
    --gradient_checkpointing true \
    --report_to tensorboard \
    --seed 45678
