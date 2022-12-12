#!/usr/bin/env bash

python train.py --image_size 1024 \
--input_size 512 \
--batch_size 32 \
--learning_rate 0.001 \
--max_epoch 100 \
--save_interval 20 \
--start_early_stopping 30 \
--early_stopping_patience 10 \
--load_from SOTA_pretrained.pth