#!/usr/bin/env bash
set -e
python -m src.cli pretrain --config configs/model_0p5b.yaml --data fineweb       --tokens 200_000_000 --batch_size 1 --grad_accum 32 --amp --bf16 --compile       --ckpt_out outputs/pretrain/final.pth --eval_every_steps 0
