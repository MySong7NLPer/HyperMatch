#!/usr/bin/env bash
export DATA_PATH=/data/nick/dataset/bertkpe_dataset

# preprocess openkp or kp20k
python preprocess.py --dataset_class inspec --source_dataset_dir $DATA_PATH/dataset --output_path $DATA_PATH/prepro_dataset

#python preprocess.py --dataset_class openkp --source_dataset_dir $DATA_PATH/dataset --output_path $DATA_PATH/prepro_dataset