#!/usr/bin/env bash
export DATA_PATH=/data/nick/dataset/bertkpe_dataset
# ** indicates your file
CUDA_VISIBLE_DEVICES=7 python test.py --run_mode test \
--local_rank -1 \
--model_class bert2rank \
--pretrain_model_type roberta-base \
--dataset_class nus \
--per_gpu_test_batch_size 64 \
--preprocess_folder $DATA_PATH/prepro_dataset \
--pretrain_model_path $DATA_PATH/pretrain_model \
--cached_features_dir $DATA_PATH/cached_features \
--eval_checkpoint /data/nick/hyper/results/train_bert2rank_kp20k_roberta_01.21_10.02/checkpoints/bert2rank.kp20k.roberta.epoch_10.checkpoint
