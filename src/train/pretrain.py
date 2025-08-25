import os, time, math, itertools, contextlib
from math import ceil
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..tokenization.bpe_tokenizer import BPETokenizer
from ..data.packed_lm import PackedLM
from ..data.streaming import stream_text
from ..model.gpt import GPT, GPTConfig, count_params
from ..utils.schedule import cosine
from ..utils.ema import EMA
from ..utils.optim import build_optimizer
from ..utils.logging import CSVLogger

torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def _make_loader(data, tok: BPETokenizer, seq_len, batch_size):
    """Windows 下固定 1 个 worker 避免 pickling 问题；Linux 可调大。"""
    ds = PackedLM(data, tok_model_path=tok.model_path, vocab_size=tok.vocab_size, seq_len=seq_len)
    if os.name == 'nt':
        num_workers = 1
    else:
        num_workers = 2
    kwargs = dict(
        batch_size=int(batch_size),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=False,
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


def _as_float(x, name):
    try:
        return float(x)
    except Exception as e:
        raise TypeError(f"[pretrain] hyperparam `{name}` expects float, got {x!r}") from e


def _as_int(x, name):
    try:
        return int(float(x))
    except Exception as e:
        raise TypeError(f"[pretrain] hyperparam `{name}` expects int, got {x!r}") from e


@torch.no_grad()
def _run_val(model: torch.nn.Module, tok: BPETokenizer, args, device) -> Tuple[float, float]:
    model.eval(); loader = _make_loader(args.val_data, tok, args.seq_len, args.batch_size)
    tot, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if args.val_batches and i >= args.val_batches: break
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if args.bf16 else torch.float16, enabled=bool(args.amp) and device.type == 'cuda'):
            logits, ce = model(x, y)
        tot += float(ce.item()); n += 1
    model.train()
    if n == 0: return float("inf"), float("inf")
    loss = tot / n; ppl = math.exp(min(20.0, loss))
    return loss, ppl


def _chunked_logsumexp(x: torch.Tensor, dim: int = -1, chunk: int = 2048) -> torch.Tensor:
    m16 = x.max(dim=dim, keepdim=True).values
    m32 = m16.float()
    V = x.size(-1)
    s32 = None
    for i in range(0, V, chunk):
        xi32 = x[..., i:i+chunk].float()
        si32 = (xi32 - m32).exp().sum(dim=-1)
        s32 = si32 if s32 is None else (s32 + si32)
    z32 = s32.log() + m32.squeeze(-1)
    return z32


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------- Tokenizer ----------
    tok = BPETokenizer(args.bpe_model, vocab_size=int(args.vocab_size))
    if tok.model_path is None:
        print("[Tokenizer] Training SentencePiece on a small sample...")
        texts = itertools.islice(stream_text(args.data), 0, 5000)
        tok.train_from_texts(texts, model_prefix="spm", vocab_size=int(args.vocab_size))
    tok.model_path = os.path.abspath(tok.model_path or "spm.model")

    # ---------- 强制类型修正 ----------
    args.n_layer       = _as_int(args.n_layer, "n_layer")
    args.n_head        = _as_int(args.n_head, "n_head")
    args.n_kv_head     = _as_int(args.n_kv_head, "n_kv_head")
    args.n_embd        = _as_int(args.n_embd, "n_embd")
    args.seq_len       = _as_int(args.seq_len, "seq_len")
    args.batch_size    = _as_int(args.batch_size, "batch_size")
    args.grad_accum    = _as_int(args.grad_accum, "grad_accum")

    args.lr            = _as_float(args.lr, "lr")
    args.min_lr        = _as_float(args.min_lr, "min_lr")
    args.weight_decay  = _as_float(args.weight_decay, "weight_decay")
    args.grad_clip     = _as_float(args.grad_clip, "grad_clip")
    args.z_loss        = _as_float(args.z_loss, "z_loss")
    args.ema_decay     = _as_float(args.ema_decay, "ema_decay")
    args.rope_scale    = _as_float(args.rope_scale, "rope_scale")
    args.dropout       = _as_float(args.dropout, "dropout")

    args.tokens            = _as_int(args.tokens, "tokens")
    args.epochs            = _as_int(args.epochs, "epochs")
    args.warmup_steps      = _as_int(args.warmup_steps, "warmup_steps")
    args.val_batches       = _as_int(args.val_batches, "val_batches") if args.val_batches is not None else None
    args.eval_every_steps  = _as_int(args.eval_every_steps, "eval_every_steps") if args.eval_every_steps is not None else 0
    args.early_stop_patience = _as_int(args.early_stop_patience, "early_stop_patience") if args.early_stop_patience is not None else 0

    # ---------- Data ----------
    loader = _make_loader(args.data, tok, args.seq_len, args.batch_size)

    # ---------- Model ----------
    cfg = GPTConfig(
        vocab_size=int(args.vocab_size), n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head,
        n_embd=args.n_embd, seq_len=args.seq_len, rope_scale=args.rope_scale, rope_mode=args.rope_mode,
        qk_norm=bool(args.qk_norm), dropout=args.dropout, checkpoint=bool(args.checkpoint)
    )
    model = GPT(cfg).to(device)
    if getattr(args, "compile", False):
        try:
            model = torch.compile(model)
        except Exception as e:
            print("[Warn] torch.compile failed:", e)

    print(f"[Model] params={count_params(model)/1e6:.1f}M  n_layer={cfg.n_layer} n_head={cfg.n_head} d_model={cfg.n_embd}")

    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp) and device.type == 'cuda' and not bool(args.bf16))
    opt = build_optimizer(model, lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay, fused=True)
    ema = EMA(model, decay=args.ema_decay) if bool(getattr(args, "ema", False)) else None

    step = 0; accum = args.grad_accum
    toks_per_step = args.batch_size * args.seq_len * accum
    total_steps = ceil(args.tokens / toks_per_step) if args.tokens > 0 else args.epochs * 10000
    print(f"[Train] tokens_budget={args.tokens}  toks/step={toks_per_step}  total_steps={total_steps}")

    pbar = tqdm(total=total_steps, desc="pretrain", dynamic_ncols=True)
    t0 = time.time(); tokens_done = 0
    os.makedirs(os.path.dirname(args.ckpt_out), exist_ok=True)
    logger = CSVLogger(os.path.join(os.path.dirname(args.ckpt_out), "train_log.csv"),
                       fieldnames=["step","loss","ce","zreg","lr","tok_s","tokens_M"])

    best_val = float("inf"); no_improve = 0
    eval_enabled = (args.val_data is not None) and (args.eval_every_steps > 0)

    for epoch in range(args.epochs if args.tokens == 0 else 10**9):
        for it, (x, y) in enumerate(loader):
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if args.bf16 else torch.float16, enabled=bool(args.amp) and device.type == 'cuda'):
                logits, ce_loss = model(x, y)

            # -------- z-loss：分块 + 每块升 FP32，稳定 & 省显存 --------
            z_reg = torch.tensor(0.0, device=x.device, dtype=ce_loss.dtype)
            if args.z_loss > 0.0:
                # 这里不启用 autocast，helper 内部已经针对 chunk 做了 .float()
                z = _chunked_logsumexp(logits, dim=-1, chunk=2048)     # [B,T] fp32
                z_reg = (z.to(logits.dtype) ** 2).mean().to(ce_loss.dtype) * args.z_loss

            # 首步打印一次，确认 z 值正常（~10.x）
            if step == 0:
                with torch.no_grad():
                    print(f"[debug] z_mean={z.mean().item() if args.z_loss>0 else 0:.3f}, "
                          f"logits_max={logits.max().float().item():.3f}, "
                          f"ce={ce_loss.item():.3f}")

            # 梯度缩放/累积
            loss = (ce_loss + z_reg) / accum
            if scaler.is_enabled(): scaler.scale(loss).backward()
            else: loss.backward()

            if (it + 1) % accum == 0:
                lr_now = cosine(step, total_steps, args.warmup_steps, args.lr, args.min_lr)
                for pg in opt.param_groups: pg["lr"] = lr_now
                if scaler.is_enabled(): scaler.unscale_(opt)
                if args.grad_clip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if scaler.is_enabled(): scaler.step(opt); scaler.update()
                else: opt.step()
                opt.zero_grad(set_to_none=True); step += 1
                if ema is not None: ema.update()

                tokens_done += toks_per_step
                elapsed = max(time.time() - t0, 1e-9); tok_per_s = tokens_done / elapsed
                total_loss = (ce_loss + z_reg).item()
                pbar.update(1)
                pbar.set_postfix(loss=f"{total_loss:.3f}", lr=f"{lr_now:.2e}", tok_s=f"{tok_per_s:,.0f}", tokens=f"{tokens_done/1e6:.2f}M")
                logger.log(step=step, loss=total_loss, ce=ce_loss.item(), zreg=float(z_reg.item()) if args.z_loss>0 else 0.0,
                           lr=lr_now, tok_s=int(tok_per_s), tokens_M=round(tokens_done/1e6,2))

                if eval_enabled and (step % args.eval_every_steps == 0):
                    val_loss, val_ppl = _run_val(model, tok, args, device)
                    improved = val_loss < best_val - 1e-5
                    best_val = min(best_val, val_loss)
                    no_improve = 0 if improved else (no_improve + 1)
                    pbar.write(f"[val] step={step}  val_loss={val_loss:.4f}  ppl≈{val_ppl:.1f}  {'*best*' if improved else ''}")
                    if bool(getattr(args, "save_best", False)) and improved:
                        torch.save({"model": (ema or EMA(model)).shadow if ema else model.state_dict(),
                                    "cfg": cfg.__dict__, "tok": tok.model_path}, args.best_ckpt_out)
                        pbar.write(f"[val] saved best to {args.best_ckpt_out}")
                    if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience:
                        pbar.write("[EarlyStop] patience reached; stopping."); break

                if step % 2000 == 0:
                    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "tok": tok.model_path}, args.ckpt_out)
                if step >= total_steps: break
        if step >= total_steps: break
        if args.early_stop_patience > 0 and no_improve >= args.early_stop_patience: break

    if ema is not None: ema.apply_to(model)
    torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "tok": tok.model_path}, args.ckpt_out)
    pbar.close()
    final_elapsed = max(time.time() - t0, 1e-9)
    print(f"Saved checkpoint to {args.ckpt_out}")
    print(f"[Throughput] tokens_done={tokens_done:,}  elapsed={final_elapsed:.1f}s  tok/s≈{tokens_done/final_elapsed:,.0f}")
    logger.close()
