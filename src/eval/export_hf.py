import os, json, shutil
import torch
import safetensors.torch as st
from ..model.gpt import GPT, GPTConfig

def run(args):
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = GPTConfig(**ckpt["cfg"]); model = GPT(cfg); model.load_state_dict(ckpt["model"], strict=True)
    os.makedirs(args.out_dir, exist_ok=True)
    rope_scaling = None
    if cfg.rope_mode in ("yarn","ntk") and abs(cfg.rope_scale-1.0)>1e-6:
        rope_scaling = {"type": cfg.rope_mode, "factor": float(cfg.rope_scale), "original_max_position_embeddings": int(cfg.seq_len)}
    hf_cfg = {
        "_name_or_path": "", "architectures": ["LlamaForCausalLM"],
        "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        "hidden_size": cfg.n_embd, "intermediate_size": int(cfg.n_embd*4),
        "max_position_embeddings": cfg.seq_len, "num_attention_heads": cfg.n_head,
        "num_hidden_layers": cfg.n_layer, "num_key_value_heads": cfg.n_kv_head,
        "rms_norm_eps": 1e-6, "rope_theta": float(cfg.rope_base), "rope_scaling": rope_scaling,
        "vocab_size": cfg.vocab_size, "tie_word_embeddings": True, "torch_dtype": "bfloat16"
    }
    with open(os.path.join(args.out_dir,"config.json"),"w") as f: json.dump(hf_cfg,f,indent=2)
    sd = ckpt["model"]; out = {}
    out["model.embed_tokens.weight"]=sd["tok_emb.weight"].clone()
    H=cfg.n_head; Hd=cfg.n_embd//cfg.n_head; K=cfg.n_kv_head
    for i in range(cfg.n_layer):
        p=f"blocks.{i}."; qkv=sd[p+"attn.qkv.weight"]
        out[f"model.layers.{i}.self_attn.q_proj.weight"]=qkv[:H*Hd,:]
        out[f"model.layers.{i}.self_attn.k_proj.weight"]=qkv[H*Hd:H*Hd+K*Hd,:]
        out[f"model.layers.{i}.self_attn.v_proj.weight"]=qkv[H*Hd+K*Hd:,:]
        out[f"model.layers.{i}.self_attn.o_proj.weight"]=sd[p+"attn.o_proj.weight"]
        out[f"model.layers.{i}.input_layernorm.weight"]=sd[p+"n1.weight"]
        out[f"model.layers.{i}.post_attention_layernorm.weight"]=sd[p+"n2.weight"]
        out[f"model.layers.{i}.mlp.gate_proj.weight"]=sd[p+"mlp.w1.weight"]
        out[f"model.layers.{i}.mlp.up_proj.weight"]=sd[p+"mlp.w2.weight"]
        out[f"model.layers.{i}.mlp.down_proj.weight"]=sd[p+"mlp.w3.weight"]
    out["model.norm.weight"]=sd["norm.weight"]; out["lm_head.weight"]=sd["lm_head.weight"].clone()
    st.save_file(out, os.path.join(args.out_dir,"model.safetensors"))
    tok_path = ckpt.get("tok", None)
    if tok_path is None or not os.path.exists(tok_path):
        raise RuntimeError("Tokenizer path missing in checkpoint; re-run pretrain to create spm.model.")
    shutil.copy(tok_path, os.path.join(args.out_dir,"tokenizer.model"))
    with open(os.path.join(args.out_dir,"tokenizer_config.json"),"w") as f:
        json.dump({"unk_token":"<unk>","bos_token":"<s>","eos_token":"</s>","pad_token":"<pad>","model_max_length":cfg.seq_len}, f, indent=2)
    with open(os.path.join(args.out_dir,"generation_config.json"),"w") as f:
        json.dump({"max_new_tokens":128,"do_sample":False,"temperature":0.0}, f, indent=2)
    print("Exported HF model to", args.out_dir)
