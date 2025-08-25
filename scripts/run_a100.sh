#!/usr/bin/env bash
set -e
python -m src.cli pretrain --config configs/model_1b.yaml --data fineweb       --tokens 2_000_000_000 --batch_size 8 --grad_accum 16 --amp --bf16       --ckpt_out outputs/pretrain/final.pth --eval_every_steps 2000 --val_data fineweb-edu
