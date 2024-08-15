#!/bin/bash

# 确保输出目录存在
# mkdir -p ./5.5output
mkdir -p ./5.10student_output_crd

# 定义学生模型列表
models=("resnet8" "resnet14" "resnet20" "resnet32" "resnet44" "resnet56" "resnet110" "resnet8x4" "resnet32x4" "wrn_16_1" "wrn_16_2" "wrn_40_1" "wrn_40_2" "vgg8" "vgg11" "vgg13" "vgg16" "vgg19" "ResNet50" "MobileNetV2" "ShuffleV1" "ShuffleV2")

# 定义学习率和衰减因子
learning_rates=(0.1 0.01 0.001)
decay_factors=(0.0005 0.0001)

# 循环遍历所有模型和参数组合
for model in "${models[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for decay in "${decay_factors[@]}"; do
            # 使用知识蒸馏策略
            echo "Training ${model} with LR=${lr}, Decay=${decay}"
            python train_student.py \
                --path_t ./save/models/resnet32_CICIOV2024_lr_0.05_decay_0.0005_trial_0/resnet32_CICIOV2024_best.pth \
                --distill crd \
                --model_s $model \
                -r $lr \
                -a 0.9 \
                -b $decay \
                --trial 1 \
                --dataset CICIOV2024 > ./5.10student_output_crd/${model}_LR${lr}_DECAY${decay}_CICIOV2024.txt 2>&1
            # 检查命令是否成功
            if [ $? -ne 0 ]; then
                echo "Error training ${model} with LR=${lr}, Decay=${decay}"
            fi
        done
    done
done
