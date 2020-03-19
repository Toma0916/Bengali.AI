#!/bin/bash
export CUDA_VISIBLE_DEVICES="0, 1, 2, 3"
python train.py --batch_size 64 --epochs 10 --output_dir example1 --debug