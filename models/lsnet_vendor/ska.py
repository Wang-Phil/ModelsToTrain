"""SKA (Sparse Kernel Attention): Triton CUDA（上游）与纯 PyTorch 等价实现。

环境变量:
  LSNET_SKA_TRITON=1   仅使用 Triton（失败则抛错）
  LSNET_SKA_TRITON=0   仅使用 PyTorch（较慢但稳定，可避免 ``context is destroyed`` 等 Triton/CUDA 问题）
  未设置或其它值      先尝试 Triton，任一 CUDA RuntimeError 后本会话内改用 PyTorch
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Function

try:
    import triton
    import triton.language as tl

    _TRITON_IMPORT_OK = True
except ImportError:
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_IMPORT_OK = False

try:
    from torch.amp import custom_bwd, custom_fwd
except ImportError:
    from torch.cuda.amp import custom_bwd, custom_fwd


def _ska_env_use_triton_only() -> bool | None:
    """None=auto, True=triton only, False=pytorch only."""
    v = os.environ.get("LSNET_SKA_TRITON", "").strip().lower()
    if v in ("1", "true", "yes", "triton"):
        return True
    if v in ("0", "false", "no", "pt", "torch", "pytorch"):
        return False
    return None


def ska_forward_reference(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    与 Triton ska_fwd 等价的向量化实现（支持 autograd）。
    x: (N, C, H, W), w: (N, wc, ks*ks, H, W)
    """
    n, ic, h, width = x.shape
    wc = w.shape[1]
    kk = w.shape[2]
    ks = int(math.sqrt(kk))
    pad = (ks - 1) // 2
    x_pad = F.pad(x, (pad, pad, pad, pad))
    ph = x_pad.unfold(2, ks, 1).unfold(3, ks, 1)
    ph = ph.reshape(n, ic, h, width, kk).permute(0, 1, 4, 2, 3).contiguous()
    wc_idx = torch.arange(ic, device=x.device, dtype=torch.long) % wc
    w_exp = w[:, wc_idx]
    return (ph * w_exp).sum(dim=2)


if _TRITON_IMPORT_OK:

    def _apply_fwd_decorator(fn):
        try:
            return custom_fwd(device_type="cuda")(fn)
        except TypeError:
            return custom_fwd(fn)

    def _apply_bwd_decorator(fn):
        try:
            return custom_bwd(device_type="cuda")(fn)
        except TypeError:
            return custom_bwd(fn)

    def _grid(numel: int, bs: int) -> tuple:
        return (triton.cdiv(numel, bs),)

    @triton.jit
    def _idx(i, n: int, c: int, h: int, w: int):
        ni = i // (c * h * w)
        ci = (i // (h * w)) % c
        hi = (i // w) % h
        wi = i % w
        m = i < (n * c * h * w)
        return ni, ci, hi, wi, m

    @triton.jit
    def ska_fwd(
        x_ptr,
        w_ptr,
        o_ptr,
        n,
        ic,
        h,
        w,
        ks,
        pad,
        wc,
        BS: tl.constexpr,
        CT: tl.constexpr,
        AT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
        val = tl.zeros((BS,), dtype=AT)

        for kh in range(ks):
            hin = hi - pad + kh
            hb = (hin >= 0) & (hin < h)
            for kw in range(ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)

                x_off = ((ni * ic + ci) * h + hin) * w + win
                w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

                x_val = tl.load(x_ptr + x_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, x_val * w_val, 0.0).to(AT)

        tl.store(o_ptr + offs, val.to(CT), mask=m)

    @triton.jit
    def ska_bwd_x(
        go_ptr,
        w_ptr,
        gi_ptr,
        n,
        ic,
        h,
        w,
        ks,
        pad,
        wc,
        BS: tl.constexpr,
        CT: tl.constexpr,
        AT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, ic, h, w)
        val = tl.zeros((BS,), dtype=AT)

        for kh in range(ks):
            ho = hi + pad - kh
            hb = (ho >= 0) & (ho < h)
            for kw in range(ks):
                wo = wi + pad - kw
                b = hb & (wo >= 0) & (wo < w)

                go_off = ((ni * ic + ci) * h + ho) * w + wo
                w_off = ((ni * wc + ci % wc) * ks * ks + (kh * ks + kw)) * h * w + ho * w + wo

                go_val = tl.load(go_ptr + go_off, mask=m & b, other=0.0).to(CT)
                w_val = tl.load(w_ptr + w_off, mask=m, other=0.0).to(CT)
                val += tl.where(b & m, go_val * w_val, 0.0).to(AT)

        tl.store(gi_ptr + offs, val.to(CT), mask=m)

    @triton.jit
    def ska_bwd_w(
        go_ptr,
        x_ptr,
        gw_ptr,
        n,
        wc,
        h,
        w,
        ic,
        ks,
        pad,
        BS: tl.constexpr,
        CT: tl.constexpr,
        AT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        start = pid * BS
        offs = start + tl.arange(0, BS)

        ni, ci, hi, wi, m = _idx(offs, n, wc, h, w)

        for kh in range(ks):
            hin = hi - pad + kh
            hb = (hin >= 0) & (hin < h)
            for kw in range(ks):
                win = wi - pad + kw
                b = hb & (win >= 0) & (win < w)
                w_off = ((ni * wc + ci) * ks * ks + (kh * ks + kw)) * h * w + hi * w + wi

                val = tl.zeros((BS,), dtype=AT)
                steps = (ic - ci + wc - 1) // wc
                for s in range(tl.max(steps, axis=0)):
                    cc = ci + s * wc
                    cm = (cc < ic) & m & b

                    x_off = ((ni * ic + cc) * h + hin) * w + win
                    go_off = ((ni * ic + cc) * h + hi) * w + wi

                    x_val = tl.load(x_ptr + x_off, mask=cm, other=0.0).to(CT)
                    go_val = tl.load(go_ptr + go_off, mask=cm, other=0.0).to(CT)
                    val += tl.where(cm, x_val * go_val, 0.0).to(AT)

                tl.store(gw_ptr + w_off, val.to(CT), mask=m)

    class SkaFn(Function):
        @staticmethod
        @_apply_fwd_decorator
        def forward(ctx, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            ks = int(math.sqrt(w.shape[2]))
            pad = (ks - 1) // 2
            ctx.ks, ctx.pad = ks, pad
            n, ic, h, width = x.shape
            wc = w.shape[1]
            o = torch.empty(n, ic, h, width, device=x.device, dtype=x.dtype)
            numel = o.numel()

            x = x.contiguous()
            w = w.contiguous()

            grid = lambda meta: _grid(numel, meta["BS"])

            ct = (
                tl.float16
                if x.dtype == torch.float16
                else (tl.float32 if x.dtype == torch.float32 else tl.float64)
            )
            at = tl.float32 if x.dtype == torch.float16 else ct

            ska_fwd[grid](x, w, o, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at)

            ctx.save_for_backward(x, w)
            ctx.ct, ctx.at = ct, at
            return o

        @staticmethod
        @_apply_bwd_decorator
        def backward(ctx, go: torch.Tensor) -> tuple:
            ks, pad = ctx.ks, ctx.pad
            x, w = ctx.saved_tensors
            n, ic, h, width = x.shape
            wc = w.shape[1]

            go = go.contiguous()
            gx = gw = None
            ct, at = ctx.ct, ctx.at

            if ctx.needs_input_grad[0]:
                gx = torch.empty_like(x)
                numel = gx.numel()
                ska_bwd_x[lambda meta: _grid(numel, meta["BS"])](
                    go, w, gx, n, ic, h, width, ks, pad, wc, BS=1024, CT=ct, AT=at
                )

            if ctx.needs_input_grad[1]:
                gw = torch.empty_like(w)
                numel = gw.numel() // w.shape[2]
                ska_bwd_w[lambda meta: _grid(numel, meta["BS"])](
                    go, x, gw, n, wc, h, width, ic, ks, pad, BS=1024, CT=ct, AT=at
                )

            return gx, gw


_triton_ska_runtime_failed = False


class SKA(nn.Module):
    """SKA：默认 GPU 上先试 Triton，CUDA/Triton 报错则本会话内改用 PyTorch。"""

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        mode = _ska_env_use_triton_only()

        if not x.is_cuda or not _TRITON_IMPORT_OK:
            return ska_forward_reference(x, w)

        if mode is False:
            return ska_forward_reference(x, w)

        if mode is True:
            if not _TRITON_IMPORT_OK:
                raise RuntimeError("LSNET_SKA_TRITON=1 但未安装 triton")
            return SkaFn.apply(x, w)

        global _triton_ska_runtime_failed
        if _triton_ska_runtime_failed:
            return ska_forward_reference(x, w)
        try:
            return SkaFn.apply(x, w)
        except RuntimeError as e:
            msg = str(e).lower()
            if any(k in msg for k in ("cuda", "triton", "context")):
                _triton_ska_runtime_failed = True
                return ska_forward_reference(x, w)
            raise
