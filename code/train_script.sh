#!/usr/bin/env bash
python train.py --image_size 1024 \
--input_size 512 \
--batch_size 32 \
--learning_rate 0.001 \
--max_epoch 200 \
--save_interval 10 \
--start_early_stopping 50 \
--early_stopping_patience 20