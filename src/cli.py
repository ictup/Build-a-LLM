import argparse, yaml, os
from dataclasses import asdict
from .train import pretrain as pretrain_mod
from .train import longft as longft_mod
from .eval import export_hf as export_hf_mod
from .eval import run_eval as run_eval_mod
from .model.gpt import GPTConfig

def add_common(p):
    p.add_argument("--data", type=str, default="fineweb", help="fineweb|fineweb-edu|fineweb-2|slim-pajama|hub_id|/path/to/txt_dir")
    p.add_argument("--bpe_model", type=str, default=None)
    p.add_argument("--vocab_size", type=int, default=32000)
    p.add_argument("--seq_len", type=int, default=4096)

    p.add_argument("--n_layer", type=int, default=24)
    p.add_argument("--n_head", type=int, default=20)
    p.add_argument("--n_kv_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=2048)

    p.add_argument("--rope_mode", type=str, choices=["ntk","yarn"], default="yarn")
    p.add_argument("--rope_scale", type=float, default=1.5)
    p.add_argument("--qk_norm", action="store_true")
    p.add_argument("--dropout", type=float, default=0.0)

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=32)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--tokens", type=int, default=0, help="If >0: stop by token budget")

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--z_loss", type=float, default=1e-4)
    p.add_argument("--ema", action="store_true")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--checkpoint", action="store_true")
    p.add_argument("--compile", action="store_true")

    p.add_argument("--config", type=str, default=None, help="YAML to override defaults")

def resolve_config(args):
    if args.config and os.path.isfile(args.config):
        with open(args.config) as f:
            y = yaml.safe_load(f)
        for k,v in y.items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")

    pt = sub.add_parser("pretrain")
    add_common(pt)
    pt.add_argument("--ckpt_out", type=str, default="outputs/pretrain/final.pth")
    pt.add_argument("--val_data", type=str, default=None)
    pt.add_argument("--val_batches", type=int, default=100)
    pt.add_argument("--eval_every_steps", type=int, default=0)
    pt.add_argument("--early_stop_patience", type=int, default=0)
    pt.add_argument("--save_best", action="store_true")
    pt.add_argument("--best_ckpt_out", type=str, default="outputs/pretrain/best_val.pth")

    lf = sub.add_parser("longft")
    add_common(lf)
    lf.add_argument("--ckpt_in", type=str, required=True)
    lf.add_argument("--ckpt_out", type=str, default="outputs/longft/final.pth")

    ex = sub.add_parser("export_hf")
    ex.add_argument("--ckpt", type=str, required=True)
    ex.add_argument("--out_dir", type=str, required=True)

    ev = sub.add_parser("eval")
    ev.add_argument("--model_path", type=str, required=True)
    ev.add_argument("--tasks", type=str, required=True, help="comma list or path to tasks.txt")
    ev.add_argument("--limit", type=int, default=None)

    ct = sub.add_parser("count_tokens")
    add_common(ct)
    ct.add_argument("--max_samples", type=int, default=100000)
    ct.add_argument("--estimate_rows", type=int, default=None)

    args = p.parse_args()
    if args.cmd in ("pretrain","longft","count_tokens"):
        args = resolve_config(args)

    if args.cmd == "pretrain": pretrain_mod.run(args)
    elif args.cmd == "longft": longft_mod.run(args)
    elif args.cmd == "export_hf": export_hf_mod.run(args)
    elif args.cmd == "eval": run_eval_mod.run(args)
    elif args.cmd == "count_tokens":
        from .tools.count_tokens import run as ct_run
        ct_run(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
