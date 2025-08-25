# src/model/layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, mult: float = 4.0, dropout: float = 0.0):
        super().__init__()
        inner = int(d_model * mult)
        self.w1 = nn.Linear(d_model, inner, bias=False)  # gate
        self.w2 = nn.Linear(d_model, inner, bias=False)  # up
        self.w3 = nn.Linear(inner, d_model, bias=False)  # down
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.w3(F.silu(self.w1(x)) * self.w2(x)))

class Rotary(nn.Module):
    """
    标准 RoPE：最后一维按 (even, odd) 交错旋转
    输入/输出形状: [B, H, T, D]，其中 D 必须为偶数（每个头维度）
    """
    def __init__(self, head_dim: int, base: float = 10000.0, scale: float = 1.0, mode: str = "yarn"):
        super().__init__()
        self.hd = head_dim
        self.base = base
        self.scale = scale
        self.mode = mode
        self._cache = {}  # (T, device, dtype) -> (cos, sin)

    def _cos_sin(self, T: int, device, dtype):
        key = (T, device, torch.float32)  # 先用 fp32 生成，数值更稳
        if key not in self._cache:
            theta = self.base * (self.scale if self.scale > 0 else 1.0)
            inv_freq = 1.0 / (theta ** (torch.arange(0, self.hd, 2, device=device).float() / self.hd))  # [D/2]
            t = torch.arange(T, device=device).float()  # [T]
            freqs = torch.einsum("i,j->ij", t, inv_freq)  # [T, D/2]
            # 关键：把 T 放到第 3 个维度，得到 [1,1,T,D/2]，与 [B,H,T,D/2] 对齐
            cos = freqs.cos()[None, None, :, :]  # [1,1,T,D/2]
            sin = freqs.sin()[None, None, :, :]  # [1,1,T,D/2]
            self._cache[key] = (cos, sin)
        cos, sin = self._cache[key]
        # 返回与 x 相同的 dtype，避免隐式类型转换占显存
        return cos.to(dtype=dtype), sin.to(dtype=dtype)

    def apply_rotary(self, x: torch.Tensor) -> torch.Tensor:
        """
        对形状 [B,H,T,D] 的张量施加旋转位置编码；D 必须为偶数。
        """
        B, H, T, D = x.shape
        assert D % 2 == 0, "RoPE head dim must be even"
        cos, sin = self._cos_sin(T, x.device, x.dtype)   # [1,1,T,D/2]
        x_even = x[..., ::2]                              # [B,H,T,D/2]
        x_odd  = x[..., 1::2]                             # [B,H,T,D/2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd  = x_even * sin + x_odd * cos
        # 交错还原为 [B,H,T,D]
        x_out = torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)
        return x_out
