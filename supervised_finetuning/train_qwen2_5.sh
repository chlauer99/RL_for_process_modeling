#!/bin/bash

FORCE_TORCHRUN=1 llamafactory-cli train qwen2_5_lora.yaml

llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-14B-Instruct \
    --adapter_name_or_path /home/Training/qwen2.5/lora/sft \
    --template llama3 \
    --finetuning_type lora \
    --export_dir /home/Training/qwen2.5/sft-merged \
    --export_size 2 \
    --export_legacy_format False