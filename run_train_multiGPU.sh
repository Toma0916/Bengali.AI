#!/bin/bash
export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"
python train.py --batch_size 128 --epochs 30 --output_dir example1  --mixup_ratio 0.5 --cutmix_ratio 0.5 --affine_ratio 1. --debug