#!/bin/bash
#export PYTHONPATH=/root/whisper:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1

run_dir=/STT-INTERGRATION-LLM
cd $run_dir
code_dir=/scripts

speech_encoder_path=/PATH/TO/whisper/large-v3.pt
llm_path=/PATH/TO/vietmistral-7b

output_dir=/tmp/whisper-large-vietmistral-7b-$(date +"%Y%m%d")
ckpt_path=$output_dir/stt_epoch_1_step_1000
split=vivos_dev
val_data_path=/PATH/TO/${split}.jsonl
decode_log=$ckpt_path/decode_${split}_beam4

python $code_dir/inference.py \
        hydra.run.dir=$ckpt_path \
        ++model_config.llm_name="vietmistral-7b" \
        ++model_config.llm_path=$llm_path \
        ++model_config.llm_dim=4096 \
        ++model_config.encoder_name=whisper \
        ++model_config.encoder_projector_ds_rate=5 \
        ++model_config.encoder_path=$speech_encoder_path \
        ++model_config.encoder_dim=1280 \
        ++model_config.encoder_projector=linear \
        ++dataset_config.dataset=speech_dataset \
        ++dataset_config.val_data_path=$val_data_path \
        ++dataset_config.input_type=mel \
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
