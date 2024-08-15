### Self-Distillation from the Last Mini-Batch (DLB)

This is a pytorch implementation for "Self-Distillation from the Last Mini-Batch for Consistency Regularization". The paper was accepted by CVPR 2022.

The paper is available at [https://arxiv.org/abs/2203.16172](https://arxiv.org/abs/2203.16172). 


Run `dlb.py` for the proposed self distillation method.


python dlb.py --lr 0.01 --batch_size 64 --epoch 30 --milestones 10 20 --T 0.7 --alpha 0.5
python dlb.py --lr 0.01 --batch_size 64 --epoch 30 --milestones 10 20 --T 0.7 --alpha 0.5 --dataset CICIOV2024 --classes_num 5
