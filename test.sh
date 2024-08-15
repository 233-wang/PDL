# python train_teacher.py --model resnet32x4 --dataset car > teacher_car.txt 2>&1
# python train_teacher.py --model resnet32x4 --dataset CICIOV2024 > teacher_CICIOV2024.txt 2>&1
# python train_teacher.py --model ShuffleV1 --dataset CICIOV2024 > ./4.25output/CICIOV2024_ShuffleV1.txt 2>&1
# python train_teacher.py --model ShuffleV2 --dataset CICIOV2024 > ./4.25output/CICIOV2024_ShuffleV2.txt 2>&1
python train_teacher.py --model MobileNetV2 --dataset CICIOV2024 > ./4.25output/CICIOV2024_MobileNetV2.txt 2>&1
# python train_teacher.py --model resnet8 --dataset CICIOV2024 > ./4.25output/CICIOV2024_resnet8.txt 2>&1

# python train_student.py --path_t ./save/models/resnet32x4_car_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset car > student_car.txt 2>&1
# python train_student.py --path_t ./save/models/resnet32x4_CICIOV2024_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth --distill kd --model_s resnet8x4 -r 0.1 -a 0.9 -b 0 --trial 1 --dataset CICIOV2024 > student_2024.txt 2>&1
# python train_student.py --path_t ./save/models/resnet32x4_car_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1 --dataset car > student_CRD_car.txt 2>&1
# python train_student.py --path_t ./save/models/resnet32x4_CICIOV2024_lr_0.05_decay_0.0005_trial_0/resnet32x4_best.pth --distill crd --model_s resnet8x4 -a 0 -b 0.8 --trial 1 --dataset CICIOV2024 > student_CRD_CICIOV2024.txt 2>&1