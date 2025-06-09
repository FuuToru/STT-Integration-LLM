#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1

export WANDB_API_KEY=879f22e33e82b78fdf67aa394cf6620c24340a4c

# Debugging settings for multiple GPUs
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

code_dir=$(pwd)/scripts

speech_encoder_path=nguyenvulebinh/wav2vec2-base-vietnamese-250h
llm_path=Qwen/Qwen3-1.7B
train_data_path=/kaggle/input/vivos-2025/vivos/vivos_train.jsonl
val_data_path=/kaggle/input/vivos-2025/vivos/vivos_test.jsonl
output_dir=/kaggle/working/wav2vec2-qwen3-1.7b$(date +"%Y%m%d")

hydra_args="
hydra.run.dir=$output_dir \
++model_config.llm_name=qwen3-1.7b \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=2048 \
++model_config.encoder_name=wav2vec2 \
++model_config.encoder_projector_ds_rate=5 \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=768 \
++model_config.encoder_projector=linear \
++model_config.normalize=true \
++dataset_config.normalize=true \
++dataset_config.dataset=speech_dataset \
++dataset_config.train_data_path=$train_data_path \
++dataset_config.val_data_path=$val_data_path \
++dataset_config.input_type=raw \
++dataset_config.mel_size=128 \
++train_config.model_name=asr \
++train_config.num_epochs=10 \
++train_config.freeze_encoder=true \
++train_config.freeze_llm=true \
++train_config.batching_strategy=custom \
++train_config.warmup_steps=1000 \
++train_config.total_steps=29150  \
++train_config.lr=1e-4 \
++train_config.validation_interval=1000 \
++train_config.batch_size_training=2 \
++train_config.val_batch_size=2 \
++train_config.num_workers_dataloader=2 \
++train_config.output_dir=$output_dir \
++metric=wer \
++log_config.log_file=/$output_dir/train.log \
++log_config.use_wandb=true \
++log_config.wandb_dir=$output_dir \
++log_config.wandb_entity_name=huutri231103 \
++log_config.wandb_project_name=stt \
++log_config.wandb_exp_name=wav2vec-qwen3_1.7b_$(date +"%Y%m%d") \
++log_config.log_interval=5 \
"

# -m debugpy --listen 5678 --wait-for-client
if [[ $CUDA_VISIBLE_DEVICES != *","* ]]; then
    python $code_dir/finetune.py \
        $hydra_args
else
    torchrun \
        --nnodes 1 \
        --nproc_per_node 2 \
        --master_port=29503 \
        $code_dir/finetune.py \
        ++train_config.enable_fsdp=true \
        ++train_config.enable_ddp=false \
        ++train_config.low_cpu_fsdp=true \
        ++fsdp_config.pure_bf16=true \
        $hydra_args
fi
