#!/bin/bash

# 创建输出目录
mkdir -p ./5.1output
mkdir -p ./5.1student_output

# # 训练教师模型 resnet32，并保存日志
# python train_teacher.py --model resnet32 --dataset CICIOV2024 > ./5.1output/CICIOV2024_resnet32.txt 2>&1

# 定义学生模型列表
models=("resnet8" "resnet14" "resnet20" "resnet32" "resnet44" "resnet56" "resnet110" "resnet8x4" "resnet32x4" "wrn_16_1" "wrn_16_2" "wrn_40_1" "wrn_40_2" "vgg8" "vgg11" "vgg13" "vgg16" "vgg19" "ResNet50" "MobileNetV2" "ShuffleV1" "ShuffleV2")

# 定义学习率和衰减因子
learning_rates=(0.1 0.01 0.001)
decay_factors=(0.0005 0.0001)

# 循环遍历所有模型和参数组合
for model in "${models[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for decay in "${decay_factors[@]}"; do
            # 假设使用知识蒸馏（kd）策略，可以根据需要更改
            python train_student.py \
                --path_t ./save/models/resnet32_CICIOV2024_best.pth \
                --distill kd \
                --model_s $model \
                -r $lr \
                -a 0.9 \
                -b $decay \
                --trial 1 \
                --dataset CICIOV2024 > ./student_output/${model}_LR${lr}_DECAY${decay}_CICIOV2024.txt 2>&1
        done
    done
done
