#!/bin/bash

# Default configurations
output_dir="./output/6.18student_output"
# model_path="./save/models/resnet56_CICIOV2024_lr_0.05_decay_0.0005_trial_0/resnet56_CICIOV2024_best.pth"  
model_path="./save/models/Hsnet_CICIOV2024_lr_0.05_decay_0.0005_trial_0/Hsnet_CICIOV2024_best.pth"  
# Distillation methods array
distill_methods=('kd' 'hint' 'vid' 'crd' 'kdsvd' 'nst' 'discriminatorLoss')

# Allow overriding the default output directory, model path, and distill method
while getopts "o:p:" opt; do
  case $opt in
    o) output_dir="$OPTARG";;
    p) model_path="$OPTARG";;
    \?) echo "Invalid option -$OPTARG" >&2; exit 1;;
  esac
done

# Ensure output directory exists
mkdir -p ${output_dir}

# Define the student model list
# models=("resnet8" "resnet32" "resnet8x4" "vgg8" "MobileNetV2" "ShuffleV1" "ShuffleV2" "Hsnet")
models=("Hsnet")

# Define learning rates, decay factors, and alpha values
learning_rates=(0.1)
decay_factors=(0)
alpha_values=(0.9)

# Function to train models with various distillation methods
train_model() {
    local model=$1
    local lr=$2
    local decay=$3
    local alpha=$4
    local distill=$5
    local model_output_dir="${output_dir}/${model}"
    mkdir -p ${model_output_dir}
    local trial_output="${model_output_dir}/${model}_LR${lr}_DECAY${decay}_ALPHA${alpha}_DISTILL_${distill}_CICIOV2024.txt"
    echo "Training ${model} with Distill=${distill}"
    python train_student.py \
        --path_t $model_path \
        --distill $distill \
        --model_s $model \
        -r $lr \
        -a $alpha \
        -b $decay \
        --trial 1 \
        --dataset CICIOV2024 > $trial_output 2>&1
    if [ $? -ne 0 ]; then
        echo "Error training ${model} with LR=${lr}, Decay=${decay}, Alpha=${alpha}, Distill=${distill}"
    fi
}

# Loop through all models and parameter combinations
for model in "${models[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for decay in "${decay_factors[@]}"; do
            for alpha in "${alpha_values[@]}"; do
                for distill in "${distill_methods[@]}"; do
                    train_model $model $lr $decay $alpha $distill
                done
            done
        done
    done
done
