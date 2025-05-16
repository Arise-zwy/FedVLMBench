# conda activate fedllm
# export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1。
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1  main_my.py \
#     --fed_alg "fedavg" \
#     --num_clients 1 \
#     --sample_clients 2 \
#     --model_name_or_path "/home/u3011649/Documents/Meta-Llama-3___1-8B/" \
#     --llm_type "meta-llama" \
#     --vision_tower_type clip \
#     --vision_tower "/home/u3011649/Documents/clip/" \
#     --mm_projector_type linear \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length False \
#     --bf16 True \
#     --dataset_name "scienceQA" \
#     --data_path /home/u3011649/Documents/data/ScienceQA/new_iid_split_2/ \
#     --image_folder /home/u3011649/Documents/data/ScienceQA/image \
#     --dataset_sample 20000 \
#     --pad_token_version v1 \
#     --use_peft True\
#     --lora_enable True --peft_lora_r 8 --peft_lora_alpha 16 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 1 \
#     --max_steps 10 \
#     --num_rounds 200 \
#     --batch_size 1  \
#     --load_in_8bit True\
#     --gradient_accumulation_steps 1 \
#     --output_dir "./output" \
#     --template "alpaca" \
#     --weight_decay 0. \
#     --lr_scheduler_type "cosine" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --logging_steps 1 \
#     --model_max_length 1024 \
#     --gradient_checkpointing True \
#     --bit 16 \
#     --freeze_backbone False \
#     --lazy_preprocess True \
#     --per_device_train_batch_size 1 \
#     --per_device_eval_batch_size 1 \
#     --warmup_ratio 0.03 \
#     --tf32 False \
#     --save_strategy "steps" \
#     --evaluation_strategy "no" \
#     --dataloader_num_workers 2 \
    # --report_to none \

#     # --pretrain_mm_mlp_adapter None \


export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1。
CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 main_my.py \
   
    # --max_steps 10 \


#     # --pretrain_mm_mlp_adapter None \
#     # --lora_enable True --peft_lora_r 128 --peft_lora_alpha 256 \


# export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1。
# CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes=1 main_my.py \
#     --fed_alg "fedavg" \
#     --num_clients 2 \
#     --sample_clients 2 \
#     --model_name_or_path "/home/u3011649/Documents/Meta-Llama-3___1-8B/" \
#     --llm_type "meta-llama" \
#     --vision_tower_type clip \
#     --vision_tower "/home/u3011649/Documents/clip/" \
#     --mm_projector_type linear \
#     --mm_vision_select_layer -2 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --bits 16 \
#     --dataset_name "scienceQA" \
#     --data_path /home/u3011649/Documents/data/ScienceQA/new_iid_split_2/ \
#     --image_folder /home/u3011649/Documents/data/ScienceQA/image \
#     --dataset_sample 20000 \
#     --pad_token_version v1 \
#     --use_peft True\
#     --lora_enable True --peft_lora_r 8 --peft_lora_alpha 32 \
#     --learning_rate 2e-4 \
#     --num_train_epochs 1 \
#     --max_steps 10 \
#     --num_rounds 200 \
#     --batch_size 1  \
#     --load_in_4bit False\
#     --load_in_8bit True\
#     --gradient_accumulation_steps 1 \
#     --output_dir "./output" \
#     --template "alpaca" \
#     --weight_decay 0. \
#     --lr_scheduler_type "cosine" \
#     --save_steps 50000 \
#     --save_total_limit 1 \
#     --logging_steps 1 \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \

