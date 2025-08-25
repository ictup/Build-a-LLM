# src/model/attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import Rotary

# 兼容新老 SDPA API
_USE_NEW_SDPA = False
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend  # PyTorch 2.4+
    _USE_NEW_SDPA = True
except Exception:
    try:
        from torch.backends.cuda import sdp_kernel as sdpa_kernel, SDPBackend  # old
    except Exception:
        sdpa_kernel, SDPBackend = None, None

_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")

class MHA_GQA(nn.Module):
    def __init__(self, d_model: int, n_head: int, n_kv: int, rope: Rotary, dropout: float = 0.0, qk_norm: bool = False):
        super().__init__()
        self.n_head = n_head
        self.n_kv = n_kv
        self.d_model = d_model
        self.qk_norm = qk_norm
        self.hd = d_model // n_head
        self.qkv = nn.Linear(d_model, (n_head + 2 * n_kv) * self.hd, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = rope
        self.drop = nn.Dropout(dropout)
        self.register_buffer("_mask", None, persistent=False)

    def _causal(self, T, device):
        if self._mask is not None and self._mask.size(-1) >= T:
            return self._mask[..., :T, :T]
        mask = torch.full((T, T), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        self._mask = mask
        return mask

    def forward(self, x, past_kv=None, use_cache=False):
        B, T, C = x.shape
        H = self.n_head
        Hd = self.hd
        K = self.n_kv

        qkv = self.qkv(x)
        q, k, v = torch.split(qkv, [H * Hd, K * Hd, K * Hd], dim=-1)
        q = q.view(B, T, H, Hd).transpose(1, 2)  # [B,H,T,Hd]
        k = k.view(B, T, K, Hd).transpose(1, 2)  # [B,K,T,Hd]
        v = v.view(B, T, K, Hd).transpose(1, 2)  # [B,K,T,Hd]

        # RoPE（注意：函数名改为 apply_rotary）
        q = self.rope.apply_rotary(q)
        k = self.rope.apply_rotary(k)

        # KV cache
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_kv = (k, v) if use_cache else None

        # GQA: repeat kv heads
        if K != H:
            rep = H // K
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        if self.qk_norm:
            q = F.normalize(q, p=2.0, dim=-1)
            k = F.normalize(k, p=2.0, dim=-1)

        if _HAS_SDPA and sdpa_kernel is not None:
            try:
                # 不传 mask，直接 is_causal=True 触发更省显存路径
                backends = (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH) if hasattr(SDPBackend, "EFFICIENT_ATTENTION") else (SDPBackend.MATH,)
                with sdpa_kernel(*backends):
                    out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
            except Exception:
                # 回退到显式 mask 的常规实现
                mask = self._causal(k.size(2), x.device)[None, None, -T:, :]
                scores = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)
                scores = scores + mask
                att = scores.softmax(-1)
                out = att @ v
        else:
            mask = self._causal(k.size(2), x.device)[None, None, -T:, :]
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(Hd)
            scores = scores + mask
            att = scores.softmax(-1)
            out = att @ v

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.o_proj(out)), new_kv
