#!/bin/bash

python run0shot_api.py \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --input "./data/genre_shift_sampled/test/cnndm_test_sampled.json" \
  --output "./test-output/" \
  --delay 0 \
  --token "--"
