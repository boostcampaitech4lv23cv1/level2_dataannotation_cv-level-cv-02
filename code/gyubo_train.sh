#!/usr/bin/env bash

python train.py --image_size 1024 \
--input_size 512 \
--batch_size 16 \
--learning_rate 0.001 \
--max_epoch 30 \
--save_interval 2 \
--start_early_stopping 10 \
--early_stopping_patience 10 \
--load_from "mst_25+full1719_21e.pth"