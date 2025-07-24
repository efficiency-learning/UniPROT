#!/bin/bash

ID=$((RANDOM % 90000 + 10000)) # generate 5-digit port number
export MASTER_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('', 0)); print(s.getsockname()[1]); s.close()" 2>/dev/null || echo "29500")

export header="torchrun --nproc_per_node 1 --nnodes 1 --master_port=$MASTER_PORT -m colm.train.train"

export base_training_args="--do_train True \
--max_seq_length 512 \
--use_fast_tokenizer True \
--lr_scheduler_type linear \
--warmup_ratio 0.03 \
--weight_decay 0.0 \
--logging_steps 1 \
--num_train_epochs 4 \
--bf16 False \
--tf32 False \
--fp16 True \
--overwrite_output_dir True \
--optim adamw_torch \
--seed 0 \
--percentage 1.0 \
--save_strategy epoch \
--lora_r 128 \
--lora_alpha 512 \
--lora_dropout 0.05 \
--lora_target_modules q_proj k_proj v_proj o_proj fc1 fc2 \
--learning_rate 2e-05 \
--per_device_train_batch_size 1 \
--gradient_accumulation_steps 1 \
--small_batch_ratio 0.5 \
--report_to wandb \
--data_selection_method submodlib \
--data_selection_unit mezo \
--facility_similarity cosine"
