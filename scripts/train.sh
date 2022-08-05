#!/usr/bin/env bash
export DATA_PATH=/data/nick/dataset/bertkpe_dataset
# ** indicates your file
export dataset_class=openkp # openkp , kp20k
export max_train_steps=20810 #  20810 (openkp) , 73430 (kp20k)
export model_class=bert2rank # bert2span, bert2tag, bert2chunk, bert2rank, bert2joint
export pretrain_model=roberta-base # bert-base-cased , spanbert-base-cased , roberta-base
## --------------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py --run_mode train \
--local_rank -1 \
--max_train_steps $max_train_steps \
--model_class $model_class \
--dataset_class $dataset_class \
--pretrain_model_type $pretrain_model \
--per_gpu_train_batch_size 9 \
--gradient_accumulation_steps 4 \
--per_gpu_test_batch_size 64 \
--preprocess_folder $DATA_PATH/prepro_dataset \
--pretrain_model_path $DATA_PATH/pretrain_model \
--cached_features_dir $DATA_PATH/cached_features \
--display_iter 200 \
--save_checkpoint \
--use_viso \
# --gradient_accumulation_steps 8
