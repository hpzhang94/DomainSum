#!/bin/bash

python run2shot_api.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --input "./data/genre_shift_sampled/test/cnndm_test_sampled.json" \
  --extra "./data/genre_shift_sampled/train/cnndm_train_sampled_20.json" \
  --delay 0 \
  --token "--"
