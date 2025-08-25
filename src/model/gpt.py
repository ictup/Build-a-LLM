# src/model/gpt.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from .layers import RMSNorm, SwiGLU, Rotary
from .attention import MHA_GQA

@dataclass
class GPTConfig:
    vocab_size:int=32000
    n_layer:int=24; n_head:int=20; n_kv_head:int=4; n_embd:int=2048; seq_len:int=4096
    dropout:float=0.0; rope_base:float=10000.; rope_scale:float=1.0; rope_mode:str="yarn"
    qk_norm:bool=False; tie_weights:bool=True; checkpoint:bool=False
    init_std:float=0.02  # 新增：初始化基准 std

def count_params(m:nn.Module):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

class GPT(nn.Module):
    def __init__(self, cfg:GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.rope = Rotary(cfg.n_embd // cfg.n_head, cfg.rope_base, cfg.rope_scale, cfg.rope_mode)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "n1": RMSNorm(cfg.n_embd),
                "attn": MHA_GQA(cfg.n_embd, cfg.n_head, cfg.n_kv_head, self.rope, cfg.dropout, cfg.qk_norm),
                "n2": RMSNorm(cfg.n_embd),
                "mlp": SwiGLU(cfg.n_embd, 4.0, cfg.dropout),
            }) for _ in range(cfg.n_layer)
        ])
        self.norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # 权重共享（最后做，确保 init 后绑定）
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # 残差缩放（深层稳定）
        self.resid_scale = 1.0 / math.sqrt(2.0 * cfg.n_layer)

        # 初始化：GPT-2 风格 + 残差回流投影更小 std
        self.apply(self._init_weights)
        self._init_residual_projections()

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)

    def _init_residual_projections(self):
        # 这两类投影直接回到残差流，进一步缩小初始 std 以稳住尺度
        proj_std = self.cfg.init_std / math.sqrt(self.cfg.n_layer)
        for i, blk in enumerate(self.blocks):
            # attn 的输出投影
            nn.init.normal_(blk["attn"].o_proj.weight, mean=0.0, std=proj_std)
            # mlp 的 down 投影
            nn.init.normal_(blk["mlp"].w3.weight, mean=0.0, std=proj_std)
        if not self.cfg.tie_weights:
            # 如果没绑权重，lm_head 也回流到残差（logits 决定梯度），保守起见同样缩放
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=proj_std)

    def _blk_attn(self, blk, x, pkv, use_cache):
        h, nkv = blk["attn"](blk["n1"](x), pkv, use_cache)
        return x + self.resid_scale * h, nkv  # 残差缩放

    def _blk_mlp(self, blk, x):
        return x + self.resid_scale * blk["mlp"](blk["n2"](x))  # 残差缩放

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        x = self.tok_emb(idx)
        new_kvs = [] if use_cache else None
        if self.cfg.checkpoint:
            for i, blk in enumerate(self.blocks):
                pkv = past_kv[i] if past_kv is not None else None
                x = torch.utils.checkpoint.checkpoint(
                    lambda xx: self._blk_attn(blk, xx, pkv, use_cache)[0], x, use_reentrant=False
                )
                x = torch.utils.checkpoint.checkpoint(
                    lambda xx: self._blk_mlp(blk, xx), x, use_reentrant=False
                )
        else:
            for i, blk in enumerate(self.blocks):
                pkv = past_kv[i] if past_kv is not None else None
                x, nkv = self._blk_attn(blk, x, pkv, use_cache)
                x = self._blk_mlp(blk, x)
                if use_cache and new_kvs is not None:
                    new_kvs.append(nkv)

        x = self.norm(x)
        logits = self.lm_head(x)
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
            return logits, loss
        return logits, new_kvs
