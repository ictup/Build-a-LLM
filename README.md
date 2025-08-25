# mini-foundation-llm (0.5B / 1B) — Pretrain → Long-Context → Eval

A production-grade, **refactored & upgraded** version of your codebase to pretrain a ~0.5B or ~1B dense decoder LLM, extend context, export to HF format, and **benchmark on widely cited leaderboards** (MMLU/Redux/Pro, HellaSwag, ARC-C/E, Winogrande, TruthfulQA, LAMBADA, PIQA, GSM8K, MATH, HumanEval, MBPP).

- Single-GPU A100 
- Clean module split, robust arg parsing, improved optimizer & weight-decay hygiene, logging, EMA, cosine LR, z-loss, AMP/bfloat16, optional `torch.compile`.
- Tokenizer: SentencePiece Unigram + byte-fallback (BOS/EOS/PAD/UNK ids aligned with LLaMA/Gemma style).
- Attn: GQA with PyTorch SDPA/Flash path, RoPE with scaling (NTK/Yarn-style factor).
- Datasets: streaming via 🤗 `datasets` (FineWeb/FineWeb-Edu/SlimPajama or local .txt tree).


## Quick start

```bash
# 0) Python env
pip install -r requirements.txt

# 1) Pretrain a tiny run on 4060 (sanity check)
python -m src.cli pretrain --config configs/model_0p5b.yaml --data fineweb   --tokens 200_000_000 --batch_size 1 --grad_accum 32 --amp --bf16 --compile

# 2) (Optional) Long-context extension to 16k/32k
python -m src.cli longft --ckpt_in outputs/pretrain/final.pth --data fineweb   --seq_len 16384 --batch_size 1 --grad_accum 16 --amp --bf16

# 3) Export to HF format
python -m src.cli export_hf --ckpt outputs/pretrain/final.pth --out_dir outputs/hf_export

# 4) Evaluate on standard tasks (zero-/few-shot as configured)
#    MMLU, HellaSwag, ARC, Winogrande, TruthfulQA, PIQA, LAMBADA, GSM8K, MATH, HumanEval, MBPP
python -m src.cli eval --model_path outputs/hf_export --tasks configs/tasks.txt --limit 0
```

### Hardware presets

- **A100 80GB**: bump to `--batch_size 8–16`, `--grad_accum 8–16`, `--seq_len 4096–8192`, and disable checkpointing to maximize throughput.

## Baseline sizes (YAML in `configs/`)

- **0.5B**: 16 layers × d=1536, n_head=24, n_kv_head=8, seq_len=4096 → ~0.49B params.
- **1B**: 20 layers × d=2048, n_head=32, n_kv_head=8, seq_len=4096 → ~1.01B params.

Both use SwiGLU (4×), RMSNorm, GQA, RoPE (scalable), weight tying, dropout=0 by default.

## Benchmarks we report (to match big-tech model cards)

We mirror the **task set & protocols** commonly used in GPT/Qwen/Claude/Gemma reports so your numbers are immediately legible:

- **General knowledge**: MMLU (5-shot, macro), MMLU-Redux, MMLU-Pro
- **Reasoning/commonsense**: HellaSwag (10-shot), ARC-Challenge (25-shot), Winogrande (5-shot), TruthfulQA (0-shot)
- **Language**: PIQA (0-shot), LAMBADA (0-shot)
- **Math**: GSM8K (8-shot, *no CoT*), MATH (0/4-shot, *no CoT*)
- **Code**: HumanEval (0-shot pass@1), MBPP (3-shot)

### Reference small-model baselines (for context)

- **Llama‑3.2‑1B (base)**: *MMLU 32.2 (5‑shot), ARC‑C 32.8 (25‑shot), etc.* — model card shows the exact table under “Benchmarks - English Text”.  
  Source: Meta HF model card.  
- **Llama‑3.2‑1B Instruct**: *MMLU 49.3 (5‑shot)*.  
  Source: Meta HF model card.

- **Qwen2.5—0.5B / 1.5B** (instruction-tuned) selected rows:  
  *MMLU 47.5 / 60.9; HellaSwag 52.1 / 67.9; ARC‑C 35.6 / 54.7; Winogrande 56.3 / 65.0.*  
  Source: Qwen blog.

- **GPT‑4o‑mini** (small proprietary): *MMLU ≈ 82%* (vendor blog). Useful as an **upper bound** reference for small models.

- **Gemma 3**: Report details architecture (GQA, QK‑norm, local/global attention, 1B = 2T tokens) and benchmark tables for larger sizes; we use the **same task set** for comparability.

> See the “References” section at the bottom with links.

## Training recipe highlights

- **Tokenizer**: SentencePiece Unigram, byte fallback, `pad=0, bos=1, eos=2, unk=3`. `spm.model` auto-trained on a small sample if not provided.
- **Optimization**: AdamW(betas=0.9,0.95), cosine decay with warmup, z‑loss, grad‑clip, EMA (optional). Proper WD exclusions (norms/bias/embeddings).
- **Attention**: SDPA/Flash path with a fused casual mask; falls back to math attention when needed. QK‑norm optional.
- **Checkpointing**: Activation checkpoint per block; toggled via config.
- **Logging**: tqdm + CSV; prints throughput tokens/s and saves periodic checkpoints.
- **Eval**: `lm-eval` harness with reasonable defaults to match vendor protocols (shots/metrics).


## References (vendor tech cards & reports)

- Meta: Llama 3.2 1B card (benchmarks table) — https://huggingface.co/meta-llama/Llama-3.2-1B  
- Meta: Llama 3.2 evals collection — https://huggingface.co/collections/meta-llama/llama-32-evals-66f44b3d2df1c7b136d821f0  
- Alibaba: Qwen2.5 LLM blog (0.5B/1.5B/3B metrics tables) — https://qwenlm.github.io/blog/qwen2.5-llm/  
- OpenAI: GPT‑4o mini (MMLU ≈ 82%) — https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/  
- Google DeepMind: Gemma 3 Tech Report (architecture & eval setup) — https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf

---

