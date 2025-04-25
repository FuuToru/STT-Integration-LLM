#!/bin/bash
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1


code_dir=$(pwd)/scripts

speech_encoder_path=nguyenvulebinh/wav2vec2-base-vietnamese-250h
llm_path=Qwen/Qwen2.5-1.5B

output_dir=/kaggle/working
ckpt_path=$output_dir/wav2vec-qwen1.5b-vivos-20-4
split=vivos_test
val_data_path=/kaggle/input/vivos-2025/vivos/vivos_test.jsonl
decode_log=$ckpt_path/decode_${split}_beam4

python $code_dir/inference.py \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name=qwen2.5-1.5b \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=1536 \
        ++model_config.encoder_name=wav2vec2 \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=768 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=raw \
        ++dataset_config.mel_size=128 \
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=asr \
        ++train_config.freeze_encoder=true \
        ++train_config.freeze_llm=true \
        ++train_config.batching_strategy=custom \
        ++train_config.num_epochs=1 \
        ++train_config.val_batch_size=1 \
        ++train_config.num_workers_dataloader=2 \
        ++train_config.output_dir=$output_dir \
        ++decode_log=$decode_log \
        ++ckpt_path=$ckpt_path/model.pt \
