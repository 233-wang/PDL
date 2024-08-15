#!/bin/bash

# Default configuration
output_dir="./output/6.18_teacher_output"

# Define available models and datasets
# models=("resnet8" "resnet14" "resnet20" "resnet32" "resnet44" "resnet56" 
#         "resnet8x4" "resnet32x4" "wrn_16_1" "wrn_16_2" "wrn_40_1" "wrn_40_2"
#         "vgg8" "vgg11" "vgg13" "vgg16"
#         "MobileNetV2" "ShuffleV1" "ShuffleV2")
# models=("resnet8" "resnet32"  "resnet8x4" "vgg8" "MobileNetV2" "ShuffleV1" "ShuffleV2" "Hsnet")        
models=("Hsnet")

 #9*9*3 -》32*32*3       
datasets=("CICIOV2024")
# datasets=("ROAD")
# Allow overriding the default output directory
while getopts "o:" opt; do
  case $opt in
    o) output_dir="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

# Function to train teacher models for a specific dataset and manage outputs in dataset-specific subdirectories
train_teacher() {
    local model=$1
    local dataset=$2
    local dataset_dir="${output_dir}/${dataset}"  # Directory for specific dataset
    mkdir -p ${dataset_dir}  # Ensure dataset-specific directory exists
    local trial_output="${dataset_dir}/${model}.txt"
    echo "Training teacher model ${model} on dataset ${dataset}"
    python train_teacher.py \
        --model $model \
        --dataset $dataset > $trial_output 2>&1
    if [ $? -ne 0 ]; then
        echo "Error training teacher model ${model} on dataset ${dataset}"
    fi
}

# Loop through all models and datasets
for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do
        train_teacher $model $dataset
    done
done
