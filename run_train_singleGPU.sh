#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
python train.py --batch_size 4 --epochs 10 --output_dir example1 --debug --mixup_ratio 0.5 --cutmix_ratio 0.5 --affine_ratio 1.