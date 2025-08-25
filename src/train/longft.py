import os, time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..model.gpt import GPT, GPTConfig
from ..tokenization.bpe_tokenizer import BPETokenizer
from ..data.packed_lm import PackedLM
from ..utils.ema import EMA
from ..utils.optim import build_optimizer


def _make_loader(data, tok: BPETokenizer, seq_len, batch_size):
    ds = PackedLM(data, tok_model_path=tok.model_path, vocab_size=tok.vocab_size, seq_len=seq_len)

    if os.name == 'nt':
        num_workers = 1  # 关键修改
    else:
        num_workers = 2

    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,  # 更稳
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2

    return DataLoader(ds, **kwargs)


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_in, map_location="cpu")
    base_cfg = GPTConfig(**ckpt["cfg"])
    long_cfg = GPTConfig(
        vocab_size=base_cfg.vocab_size, n_layer=base_cfg.n_layer, n_head=base_cfg.n_head, n_kv_head=base_cfg.n_kv_head,
        n_embd=base_cfg.n_embd, seq_len=args.seq_len, rope_scale=args.rope_scale, rope_mode=args.rope_mode,
        qk_norm=base_cfg.qk_norm, dropout=base_cfg.dropout, checkpoint=args.checkpoint or base_cfg.checkpoint
    )
    model = GPT(long_cfg); model.load_state_dict(ckpt["model"], strict=False); model = model.to(device)
    tok = BPETokenizer(ckpt.get("tok", None), vocab_size=base_cfg.vocab_size)
    loader = _make_loader(args.data, tok, args.seq_len, args.batch_size)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type=='cuda' and not args.bf16)
    opt = build_optimizer(model, lr=args.lr, betas=(0.9,0.95), weight_decay=0.1, fused=True)
    ema = EMA(model, decay=0.999)

    step = 0; accum = args.grad_accum; total_steps = args.epochs * 2000
    pbar = tqdm(total=total_steps, desc="longft", dynamic_ncols=True)
    t0 = time.time(); tokens_done = 0; toks_per_step = args.batch_size * args.seq_len * accum

    for epoch in range(args.epochs):
        for it,(x,y) in enumerate(loader):
            x=x.to(device, non_blocking=True); y=y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if args.bf16 else torch.float16, enabled=args.amp and device.type=='cuda'):
                logits, ce_loss=model(x,y); z=torch.logsumexp(logits.float(), dim=-1); loss=(ce_loss + (z**2).mean()*1e-4)/accum
            if scaler.is_enabled(): scaler.scale(loss).backward()
            else: loss.backward()
            if (it+1)%accum==0:
                if scaler.is_enabled(): scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if scaler.is_enabled(): scaler.step(opt); scaler.update()
                else: opt.step()
                opt.zero_grad(set_to_none=True); step+=1; ema.update()
                tokens_done += toks_per_step; tok_per_s = tokens_done / max(time.time()-t0,1e-9)
                pbar.update(1); pbar.set_postfix(loss=f"{ce_loss.item():.3f}", tok_s=f"{tok_per_s:,.0f}", tokens=f"{tokens_done/1e6:.2f}M")
                if step>=total_steps: break
        if step>=total_steps: break
    ema.apply_to(model)
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": long_cfg.__dict__, "tok": ckpt.get("tok", None)}, args.ckpt_out)
    pbar.close(); elapsed=max(time.time()-t0,1e-9)
    print("Saved long-context checkpoint to", args.ckpt_out)
    print(f"[Throughput-LongFT] tokens_done={tokens_done:,}  elapsed={elapsed:.1f}s  tok/s≈{tokens_done/elapsed:,.0f}")
