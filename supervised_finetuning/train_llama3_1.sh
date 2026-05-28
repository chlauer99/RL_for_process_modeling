#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train llama3_1_lora.yaml

llamafactory-cli export \
    --model_name_or_path meta-llama/Llama-3.1-8B-Instruct \
    --adapter_name_or_path /home/Training/llama3.1/lora/sft \
    --template llama3 \
    --finetuning_type lora \
    --export_dir /home/Training/llama3.1/sft-merged \
    --export_size 2 \
    --export_legacy_format False