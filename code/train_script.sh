#!/usr/bin/env bash
python train.py --image_size 1024 \
--input_size 512 \
--batch_size 32 \
--learning_rate 0.001 \
--max_epoch 150 \
--save_interval 5 \
--start_early_stopping 40 \
--early_stopping_patience 10