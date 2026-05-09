from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum


def create_coordinate(h, w, start=0.0, end=1.0, device=None, dtype=None):
    x = torch.linspace(start, end, h, device=device, dtype=dtype)
    y = torch.linspace(start, end, w, device=device, dtype=dtype)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return torch.stack((xx, yy), -1).view(1, h * w, 2)


def compute_basis_size(order, mirror):
    return ((order + 1) * (order + 2)) // (1 if mirror else 2)


def herme_vander_torch(z, m):
    he0 = z.new_ones(z.shape)
    if m == 0:
        return he0[:, None]
    values = [he0, z]
    for n in range(1, m):
        values.append(z * values[-1] - n * values[-2])
    return torch.stack(values, 1)


def gauss_deriv(max_order, device, dtype, kernel_size, sigma=None, include_negations=False, scale_magnitude=True):
    sigma = (kernel_size // 2) / 1.645 if sigma is None else sigma
    if kernel_size % 2 == 0:
        raise ValueError("ksize must be odd")
    half = kernel_size // 2
    x = torch.arange(-half, half + 1, dtype=dtype, device=device)
    z = x / sigma
    g = torch.exp(-0.5 * z ** 2) / (sigma * (2.0 * torch.pi) ** 0.5)
    he = herme_vander_torch(z, max_order)
    derivs_1d = [
        (((-1) ** n) / (sigma ** n) if scale_magnitude else (-1) ** n) * he[:, n] * g
        for n in range(max_order + 1)
    ]
    bank = []
    for order in range(max_order + 1):
        for i in range(order + 1):
            kernel = torch.outer(derivs_1d[order - i], derivs_1d[i])
            bank.append(kernel)
            if include_negations:
                bank.append(-kernel)
    return torch.stack(bank, 0)


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        num_groups=8,
        pad_mode="zeros",
        norm_fn=None,
        activation_fn=nn.SiLU,
        use_conv_shortcut=False,
    ):
        super().__init__()
        norm = (lambda channels: norm_fn(num_groups, channels)) if norm_fn else (lambda channels: nn.Identity())
        padding = kernel_size // 2
        self.block = nn.Sequential(
            norm(in_channels),
            activation_fn(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, padding_mode=pad_mode, bias=False),
            norm(out_channels),
            activation_fn(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, padding_mode=pad_mode, bias=False),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, 1, bias=False, padding_mode=pad_mode)
            if use_conv_shortcut or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class LearnedFeatureUnification(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int = 3, init_gaussian_derivatives: bool = False):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if init_gaussian_derivatives:
            order = 0
            while compute_basis_size(order, False) < out_channels:
                order += 1
            self.basis = nn.Parameter(
                gauss_deriv(
                    order,
                    device="cpu",
                    dtype=torch.float32,
                    kernel_size=kernel_size,
                    scale_magnitude=False,
                )[:out_channels, None]
            )
        else:
            self.basis = nn.Parameter(torch.randn(out_channels, 1, kernel_size, kernel_size))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = features.shape
        x = self._depthwise_conv(features, self.basis, self.kernel_size).view(
            batch, self.out_channels, channels, height, width
        )
        attn = F.softmax(x, dim=1)
        return attn.mean(dim=2)

    @staticmethod
    def _depthwise_conv(features, basis, kernel_size):
        _, channels, height, width = features.shape
        padding = kernel_size // 2
        x = F.pad(features, (padding, padding, padding, padding), value=0)
        x = F.conv2d(x, basis.repeat(channels, 1, 1, 1), groups=channels)
        mask = torch.ones(1, 1, height, width, dtype=x.dtype, device=x.device)
        denom = F.conv2d(
            F.pad(mask, (padding, padding, padding, padding), value=0),
            torch.ones(1, 1, kernel_size, kernel_size, dtype=x.dtype, device=x.device),
        )
        return x / denom


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RoPE(nn.Module):
    def __init__(self, dim: int, theta: int = 100):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.freqs = nn.Parameter(torch.empty(2, self.dim))

    def _device_weight_init(self):
        freqs_1d = self.theta ** torch.linspace(0, -1, self.dim // 4)
        freqs_1d = torch.cat([freqs_1d, freqs_1d])
        freqs_2d = torch.zeros(2, self.dim)
        freqs_2d[0, : self.dim // 2] = freqs_1d
        freqs_2d[1, -self.dim // 2 :] = freqs_1d
        self.freqs.data.copy_(freqs_2d * 2 * torch.pi)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        angle = coords @ self.freqs
        return x * angle.cos() + rotate_half(x) * angle.sin()


def window2d(low_res: int | Tuple[int, int], high_res: int | Tuple[int, int], ratio: float, *, device="cpu"):
    if isinstance(high_res, int):
        high_h = high_w = high_res
    else:
        high_h, high_w = high_res
    if isinstance(low_res, int):
        low_h = low_w = low_res
    else:
        low_h, low_w = low_res

    row_pos = (torch.arange(high_h, device=device, dtype=torch.float32) + 0.5) / high_h
    col_pos = (torch.arange(high_w, device=device, dtype=torch.float32) + 0.5) / high_w
    pos_r, pos_c = torch.meshgrid(row_pos, col_pos, indexing="ij")

    r_lo = (pos_r - ratio).clamp(0.0, 1.0)
    r_hi = (pos_r + ratio).clamp(0.0, 1.0)
    c_lo = (pos_c - ratio).clamp(0.0, 1.0)
    c_hi = (pos_c + ratio).clamp(0.0, 1.0)

    r0 = (r_lo * low_h).floor().long()
    r1 = (r_hi * low_h).ceil().long()
    c0 = (c_lo * low_w).floor().long()
    c1 = (c_hi * low_w).ceil().long()
    return torch.stack([r0, r1, c0, c1], dim=2)


@torch.jit.ignore
@torch.no_grad
def compute_attention_mask(high_res_h, high_res_w, low_res_h, low_res_w, window_size_ratio, device="cpu"):
    windows = window2d(
        low_res=(low_res_h, low_res_w),
        high_res=(high_res_h, high_res_w),
        ratio=window_size_ratio,
        device=device,
    )
    query_count = high_res_h * high_res_w
    r0 = windows[..., 0].reshape(query_count, 1)
    r1 = windows[..., 1].reshape(query_count, 1)
    c0 = windows[..., 2].reshape(query_count, 1)
    c1 = windows[..., 3].reshape(query_count, 1)
    rows = torch.arange(low_res_h, device=device)
    cols = torch.arange(low_res_w, device=device)
    row_ok = (rows >= r0) & (rows < r1)
    col_ok = (cols >= c0) & (cols < c1)
    attention_mask = (row_ok.unsqueeze(2) & col_ok.unsqueeze(1)).reshape(query_count, low_res_h * low_res_w)
    return ~attention_mask


class CrossAttention(nn.Module):
    def __init__(self, qk_dim, num_heads, q_chunk_size: Optional[int] = None, store_attn: bool = False):
        super().__init__()
        self.norm_q = nn.RMSNorm(qk_dim)
        self.norm_k = nn.RMSNorm(qk_dim)
        self.q_chunk_size = q_chunk_size
        self.store_attn = store_attn
        self.attention = nn.MultiheadAttention(
            embed_dim=qk_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True,
        )

    @torch.no_grad()
    def _slice_mask(self, mask, start, end):
        if mask is None:
            return None
        if mask.dim() == 2:
            return mask[start:end, :]
        if mask.dim() == 3:
            return mask[:, start:end, :]
        raise ValueError("attn_mask must be 2D or 3D")

    def forward(self, query, key, value, mask=None, q_chunk_size: Optional[int] = None, store_attn: Optional[bool] = None):
        q_chunk_size = self.q_chunk_size if q_chunk_size is None else q_chunk_size
        store_attn = self.store_attn if store_attn is None else store_attn
        val = key
        query = self.norm_q(query)
        key = self.norm_k(key)

        if q_chunk_size is None or query.size(1) <= q_chunk_size:
            _, attn = self.attention(query, key, val, average_attn_weights=True, attn_mask=mask)
            features = einsum("b i j, b j d -> b i d", attn, value)
            return features, (attn if store_attn else None)

        _, query_len, _ = query.shape
        outputs = []
        attns = [] if store_attn else None
        for start in range(0, query_len, q_chunk_size):
            end = min(start + q_chunk_size, query_len)
            q_chunk = query[:, start:end, :]
            mask_chunk = self._slice_mask(mask, start, end)
            _, attn_chunk = self.attention(q_chunk, key, val, average_attn_weights=True, attn_mask=mask_chunk)
            outputs.append(einsum("b i j, b j d -> b i d", attn_chunk, value))
            if store_attn:
                attns.append(attn_chunk)

        features = torch.cat(outputs, dim=1)
        attn_scores = torch.cat(attns, dim=1) if store_attn else None
        return features, attn_scores


class CrossAttentionBlock(nn.Module):
    def __init__(self, qk_dim, num_heads, window_ratio: float = 0.1, q_chunk_size: Optional[int] = None, **kwargs):
        super().__init__()
        self.cross_attn = CrossAttention(qk_dim, num_heads, q_chunk_size=q_chunk_size)
        self.window_ratio = window_ratio
        self.conv2d = nn.Conv2d(qk_dim, qk_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, q, k, v, q_chunk_size: Optional[int] = None, **kwargs):
        q = self.conv2d(q)
        if self.window_ratio > 0:
            attn_mask = compute_attention_mask(*q.shape[-2:], *k.shape[-2:], window_size_ratio=self.window_ratio).to(q.device)
        else:
            attn_mask = None
        batch, _, q_h, q_w = q.shape
        _, _, k_h, k_w = k.shape
        channels = v.shape[1]
        q = q.permute(0, 2, 3, 1).view(batch, q_h * q_w, -1)
        k = k.permute(0, 2, 3, 1).view(batch, k_h * k_w, -1)
        v = v.permute(0, 2, 3, 1).view(batch, k_h * k_w, -1)
        features, _ = self.cross_attn(q, k, v, mask=attn_mask, q_chunk_size=q_chunk_size)
        return features.view(batch, q_h, q_w, channels).permute(0, 3, 1, 2)


def setup_cross_attention_block(use_natten: bool, qk_dim: int, num_heads: int, window_ratio: float = 0.1, q_chunk_size: Optional[int] = None, **kwargs) -> nn.Module:
    return CrossAttentionBlock(
        qk_dim=qk_dim,
        num_heads=num_heads,
        window_ratio=window_ratio,
        q_chunk_size=q_chunk_size,
        **kwargs,
    )


class AnyUp(nn.Module):
    def __init__(
        self,
        input_dim=3,
        qk_dim=128,
        kernel_size=1,
        kernel_size_lfu=5,
        window_ratio=0.1,
        num_heads=4,
        init_gaussian_derivatives=False,
        use_natten=False,
        lfu_dim=None,
        **kwargs,
    ):
        super().__init__()
        self.qk_dim = qk_dim
        self.lfu_dim = lfu_dim if lfu_dim is not None else qk_dim
        self.window_ratio = window_ratio
        self._rb_args = dict(kernel_size=1, num_groups=8, pad_mode="reflect", norm_fn=nn.GroupNorm, activation_fn=nn.SiLU)

        self.image_encoder = self._make_encoder(input_dim, kernel_size)
        self.key_encoder = self._make_encoder(qk_dim, 1)
        self.query_encoder = self._make_encoder(qk_dim, 1)
        self.key_features_encoder = self._make_encoder(
            None,
            1,
            first_layer_k=kernel_size_lfu,
            init_gaussian_derivatives=init_gaussian_derivatives,
        )
        self.cross_decode = setup_cross_attention_block(
            use_natten=use_natten,
            qk_dim=qk_dim,
            num_heads=num_heads,
            window_ratio=window_ratio,
        )
        self.aggregation = self._make_encoder(2 * qk_dim, 3)
        self.rope = RoPE(qk_dim)
        self.rope._device_weight_init()

    def _make_encoder(self, in_ch, kernel_size, layers=2, first_layer_k=0, init_gaussian_derivatives=False):
        pre = (
            nn.Conv2d(in_ch, self.qk_dim, kernel_size, padding=kernel_size // 2, padding_mode="reflect", bias=False)
            if first_layer_k == 0
            else LearnedFeatureUnification(
                self.lfu_dim,
                first_layer_k,
                init_gaussian_derivatives=init_gaussian_derivatives,
            )
        )
        blocks = [
            ResBlock(
                self.qk_dim if first_layer_k == 0 or i != 0 else self.lfu_dim,
                self.qk_dim,
                **self._rb_args,
            )
            for i in range(layers)
        ]
        return nn.Sequential(pre, *blocks)

    @staticmethod
    def _adaptive_avg_pool_compatible(x: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
        in_h, in_w = x.shape[-2:]
        out_h, out_w = output_size
        if x.device.type == "mps":
            divisible_h = (in_h % out_h) == 0
            divisible_w = (in_w % out_w) == 0
            if not (divisible_h and divisible_w):
                return F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return F.adaptive_avg_pool2d(x, output_size=output_size)

    def upsample(self, enc_img, feats, out_size, vis_attn=False, q_chunk_size=None):
        _, _, height, width = feats.shape
        q = self._adaptive_avg_pool_compatible(self.query_encoder(enc_img), output_size=out_size)
        k = self._adaptive_avg_pool_compatible(self.key_encoder(enc_img), output_size=(height, width))
        k = torch.cat([k, self.key_features_encoder(F.normalize(feats, dim=1))], dim=1)
        k = self.aggregation(k)
        v = feats
        return self.cross_decode(q, k, v, vis_attn=vis_attn, q_chunk_size=q_chunk_size)

    def forward(self, image, features, output_size=None, vis_attn=False, q_chunk_size=None):
        output_size = output_size if output_size is not None else image.shape[-2:]
        enc = self.image_encoder(image)
        height = enc.shape[-2]
        coords = create_coordinate(height, enc.shape[-1], device=enc.device, dtype=enc.dtype)
        enc = enc.permute(0, 2, 3, 1).view(enc.shape[0], -1, enc.shape[1])
        enc = self.rope(enc, coords)
        enc = enc.view(enc.shape[0], height, -1, enc.shape[-1]).permute(0, 3, 1, 2)
        return self.upsample(enc, features, output_size, vis_attn=vis_attn, q_chunk_size=q_chunk_size)
