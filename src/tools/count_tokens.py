import os, time, json
from typing import Iterable
from ..tokenization.bpe_tokenizer import BPETokenizer
try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

def run(args):
    tok = BPETokenizer(args.bpe_model, vocab_size=args.vocab_size)
    if tok.sp is None:
        raise RuntimeError("Tokenizer not loaded. Pass --bpe_model or run pretrain once to create spm.model")
    total_tokens = 0; n = 0; t0 = time.time()

    def enc_count(text_iter, limit):
        nonlocal total_tokens, n
        for i, txt in enumerate(text_iter):
            if limit and i >= limit: break
            total_tokens += len(tok.encode(txt, add_bos=False, add_eos=True)); n += 1

    if os.path.isdir(args.data):
        def gen():
            for root,_,files in os.walk(args.data):
                for fn in files:
                    if fn.endswith(".txt"):
                        with open(os.path.join(root,fn),"r",encoding="utf-8",errors="ignore") as f: yield f.read()
        enc_count(gen(), None); mode = "exact-local"; est_total = total_tokens
    else:
        assert load_dataset is not None, "pip install datasets"
        if args.data in ("fineweb","HuggingFaceFW/fineweb-edu-score-2","fineweb-edu-score-2"):
            ds = load_dataset("HuggingFaceFW/fineweb-edu-score-2", split="train", streaming=True); key = "text"
        elif args.data in ("fineweb-edu","HuggingFaceFW/fineweb-edu"):
            ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True); key = "text"
        elif args.data in ("fineweb-2","HuggingFaceFW/fineweb-2"):
            ds = load_dataset("HuggingFaceFW/fineweb-2", split="train", streaming=True); key = "text"
        else:
            ds = load_dataset(args.data, split="train", streaming=True); first = next(iter(ds.take(1))); key = list(first.keys())[0]
        enc_count((ex[key] for ex in ds), args.max_samples); mode = "sample-stream"
        est_total = int((total_tokens/max(n,1)) * args.estimate_rows) if args.estimate_rows else total_tokens

    elapsed = max(time.time()-t0,1e-9)
    os.makedirs("outputs/stats", exist_ok=True)
    outp={"mode":mode,"data":args.data,"counted_examples":n,"tokens_counted":int(total_tokens),
          "estimate_rows":args.estimate_rows, "estimated_total_tokens":int(est_total), "seconds":elapsed}
    with open("outputs/stats/token_count.json","w") as f: json.dump(outp,f,indent=2,ensure_ascii=False)
    print(json.dumps(outp, indent=2, ensure_ascii=False))
