#!/usr/bin/env bash

python test.py \
  --ckpt /mnt/SSD8T/home/xwz/lt/DPGF-Net/checkpoints/AGIQA3k/alignment/AGIQA3k_Alignment_Task_best.pth \
  --image ./example.jpg \
  --prompt "a cat sitting on a wooden chair" \
  --task_type alignment \
  --device cuda:0 \
  --reiqa_root ./ReIQA_main
