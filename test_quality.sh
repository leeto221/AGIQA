#!/usr/bin/env bash

python test.py \
  --ckpt ./checkpoints/AGIQA3k/quality/AGIQA3k_Quality_Task_best.pth \
  --image ./example.jpg \
  --prompt "a cat sitting on a wooden chair" \
  --task_type quality \
  --device cuda:0 \
  --reiqa_root ./ReIQA_main
