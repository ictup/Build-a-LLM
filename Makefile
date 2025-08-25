    .PHONY: pretrain longft export eval

    pretrain:
	python -m src.cli pretrain --config configs/model_0p5b.yaml --data fineweb --tokens 200_000_000 --amp --bf16

    longft:
	python -m src.cli longft --ckpt_in outputs/pretrain/final.pth --data fineweb --seq_len 16384 --amp --bf16

    export:
	python -m src.cli export_hf --ckpt outputs/pretrain/final.pth --out_dir outputs/hf_export

    eval:
	python -m src.cli eval --model_path outputs/hf_export --tasks configs/tasks.txt
