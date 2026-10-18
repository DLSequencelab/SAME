"""SAME: source-preserving multibranch additive classification.

The four representation sources have fixed, separately parameterized transforms.
All available branches are evaluated; there is no sparse expert routing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SAMEConfig:
    channels: int
    classes: int
    architecture_version: str = "ordered_multiscale_v2"
    patch_length: int = 16
    patch_stride: int = 8
    width: int = 64
    patch_mode: str = "auto"
    joint_channel_threshold: int = 16
    channel_dim: int = 16
    fft_bands: int = 4
    fft_width: int = 32
    evidence_dim: int = 64
    expert_dim: int = 32
    short_kernel: int = 3
    long_kernel: int = 3
    long_dilations: tuple[int, ...] = (1, 2, 4)
    router_temperature: float = 1.0
    route_floor: float = 0.05
    dropout: float = 0.2
    normalize_input: bool = True
    position_mode: str = "sinusoidal"
    position_scale: float = 0.1
    classifier_mode: str = "linear"
    cosine_scale: float = 10.0
    pooling_mix: float = 0.5
    branch_loss_start: float = 0.1
    branch_loss_end: float = 0.02

    def __post_init__(self):
        if self.architecture_version != "ordered_multiscale_v2":
            raise ValueError("Expected the manuscript's ordered_multiscale_v2 architecture")
        positive = ("channels", "classes", "patch_length", "patch_stride", "width",
                    "joint_channel_threshold", "channel_dim", "fft_bands", "fft_width",
                    "evidence_dim", "expert_dim", "short_kernel", "long_kernel")
        for name in positive:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.classes < 2:
            raise ValueError("At least two classes are required")
        if self.patch_length < 2 or self.patch_stride > self.patch_length:
            raise ValueError("Require patch_length >= 2 and stride <= patch_length")
        if self.fft_bands > self.patch_length // 2:
            raise ValueError("fft_bands exceeds the number of non-DC FFT bins")
        if self.patch_mode not in {"auto", "joint", "factorized"}:
            raise ValueError("patch_mode must be auto, joint, or factorized")
        if any(k % 2 == 0 for k in (self.short_kernel, self.long_kernel)):
            raise ValueError("Temporal convolution kernels must be odd")
        if not self.long_dilations or any(
                not isinstance(d, int) or isinstance(d, bool) or d < 1
                for d in self.long_dilations):
            raise ValueError("long_dilations must contain positive integers")
        for name in ("router_temperature", "cosine_scale"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("route_floor", "pooling_mix", "dropout"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if self.dropout == 1:
            raise ValueError("dropout must be less than 1")
        for name in ("position_scale", "branch_loss_start", "branch_loss_end"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.position_mode not in {"none", "sinusoidal"}:
            raise ValueError("position_mode must be none or sinusoidal")
        if self.classifier_mode not in {"linear", "cosine"}:
            raise ValueError("classifier_mode must be linear or cosine")


def masked_softmax(scores: Tensor, valid: Tensor, dim: int) -> Tensor:
    value = torch.softmax(scores.float().masked_fill(~valid, -1e9), dim=dim)
    value = value * valid.float()
    return value / value.sum(dim, keepdim=True).clamp_min(1e-8)


class TemporalBlock(nn.Module):
    """One residual depthwise-separable convolution along the PATCH axis."""
    def __init__(self, width, kernel, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, kernel, groups=width, bias=False,
                                  dilation=dilation, padding=dilation * (kernel - 1) // 2)
        self.pointwise = nn.Conv1d(width, width, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        mask = valid[..., None].to(x.dtype)
        update = (self.norm(x) * mask).transpose(1, 2)
        update = self.pointwise(self.depthwise(update)).transpose(1, 2)
        return (x + self.dropout(F.gelu(update))) * mask


class BranchTransform(nn.Module):
    """Source-specific residual bottleneck transform."""
    def __init__(self, width, hidden, dropout):
        super().__init__()
        self.adapter = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, hidden),
                                     nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, width))

    def forward(self, x):
        return x + self.adapter(x)


class LocalFFT(nn.Module):
    def __init__(self, channels, length, bands, width):
        super().__init__()
        self.bands = bands
        # Length two has an all-zero symmetric Hann window.
        window = torch.hann_window(length, periodic=False) if length > 2 else torch.ones(length)
        self.register_buffer("window", window, persistent=False)
        self.project = nn.Linear(channels * bands, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, patches, valid):
        x = patches.float()
        weights = (valid.float() * self.window)[..., None, :]
        mean = (x * weights).sum(-1, keepdim=True) / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        spectrum = torch.fft.rfft((x - mean) * weights, dim=-1, norm="ortho")
        power = spectrum[..., 1:].abs().square() / weights.square().sum(-1, keepdim=True).clamp_min(1e-8)
        bands = torch.stack([v.mean(-1) for v in torch.tensor_split(power, self.bands, -1)], -1)
        return F.gelu(self.norm(self.project(torch.log1p(bands).flatten(2))))


class SAME(nn.Module):
    branch_names = ("state", "short", "long", "frequency")

    def __init__(self, config: SAMEConfig):
        super().__init__()
        self.config = config
        self.training_progress = 0.0
        factorized = config.patch_mode == "factorized" or (
            config.patch_mode == "auto" and config.channels > config.joint_channel_threshold)
        self.effective_patch_mode = "factorized" if factorized else "joint"
        c = min(config.channel_dim, config.channels) if factorized else config.channels
        self.channel_projection = nn.Linear(config.channels, c, bias=False) if factorized else nn.Identity()
        self.patch_embedding = nn.Linear(config.patch_length * c, config.width)
        self.evidence_projection = nn.Linear(config.width, config.evidence_dim)
        d = config.evidence_dim
        self.short_conv = TemporalBlock(d, config.short_kernel, 1, config.dropout)
        self.long_blocks = nn.ModuleList([
            TemporalBlock(d, config.long_kernel, dilation, config.dropout)
            for dilation in config.long_dilations])
        self.local_fft = LocalFFT(config.channels, config.patch_length, config.fft_bands, config.fft_width)
        self.frequency_projection = nn.Linear(config.fft_width, d)
        self.route_heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1)) for _ in range(4)])
        self.experts = nn.ModuleList([BranchTransform(d, config.expert_dim, config.dropout) for _ in range(4)])
        self.pool_heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(4)])
        self.classifier = nn.Linear(d, config.classes, bias=False)

    def set_training_progress(self, progress: float):
        self.training_progress = min(1.0, max(0.0, float(progress)))

    @staticmethod
    def _normalize(x, mask):
        weight = mask[..., None].to(x.dtype)
        count = weight.sum(1, keepdim=True).clamp_min(1)
        mean = (x * weight).sum(1, keepdim=True) / count
        variance = ((x - mean).square() * weight).sum(1, keepdim=True) / count
        return (x - mean) / variance.clamp_min(1e-6).sqrt() * weight

    def _patchify(self, x, mask):
        length, stride = self.config.patch_length, self.config.patch_stride
        times = torch.arange(mask.size(1), device=mask.device)
        valid_end = torch.where(mask, times + 1, 0).amax(1)
        sample_count = 1 + torch.div((valid_end - length).clamp_min(0) + stride - 1,
                                     stride, rounding_mode="floor")
        count = max(1, 1 + math.ceil(max(0, x.size(1) - length) / stride))
        right = (count - 1) * stride + length - x.size(1)
        x = F.pad(x.transpose(1, 2), (0, right)).transpose(1, 2)
        mask = F.pad(mask, (0, right), value=False)
        index = torch.arange(count, device=x.device)
        inside = mask.unfold(1, length, stride) & (index[None, :, None] < sample_count[:, None, None])
        patches = x.unfold(1, length, stride) * inside[:, :, None]
        return patches, inside, index * stride

    def _positions(self, mask, inside, starts):
        times = torch.arange(mask.size(1), device=mask.device)
        first = torch.where(mask, times, mask.size(1)).amin(1)
        last = torch.where(mask, times, 0).amax(1)
        center = starts[None] + (inside * torch.arange(inside.size(-1), device=mask.device)).sum(-1) / inside.sum(-1).clamp_min(1)
        position = ((center - first[:, None]) / (last - first).clamp_min(1)[:, None]).clamp(0, 1)
        fraction = inside.float().mean(-1)
        d = self.config.evidence_dim
        frequencies = torch.exp(torch.linspace(0, math.log(4), (d + 3) // 4, device=mask.device))
        phase = 2 * math.pi * torch.stack((position, fraction), -1)[..., None] * frequencies
        code = torch.stack((phase.sin(), phase.cos()), -1).flatten(2)[..., :d]
        scale = self.config.position_scale if self.config.position_mode == "sinusoidal" else 0.0
        return position, code * scale * inside.any(-1)[..., None]

    def _classify(self, v):
        if self.config.classifier_mode == "linear":
            return self.classifier(v)
        return self.config.cosine_scale * F.linear(F.normalize(v.float(), dim=-1),
                                                   F.normalize(self.classifier.weight.float(), dim=-1))

    def forward(self, x: Tensor, mask: Tensor | None = None, return_aux=False):
        c = self.config
        if x.ndim != 3 or x.size(-1) != c.channels:
            raise ValueError("x must have shape [batch, time, channels]")
        mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device) if mask is None else mask.bool()
        if mask.shape != x.shape[:2] or not mask.any(1).all():
            raise ValueError("Every sample requires a matching, nonempty time mask")
        # Remove padding before computing any sample-wise statistics.
        x = torch.where(mask[..., None], x.float(), 0.0)
        x = self._normalize(x, mask) if c.normalize_input else x
        patches, inside, starts = self._patchify(x, mask)
        valid = inside.any(-1)
        channel_features = self.channel_projection(patches.transpose(-1, -2))
        h = self.evidence_projection(self.patch_embedding(channel_features.flatten(2))) * valid[..., None]
        position, code = self._positions(mask, inside, starts)

        # MRE: four aligned representation sources on the same patch grid.
        state = (h + code) * valid[..., None]
        short = self.short_conv(state, valid)
        long = state
        for block in self.long_blocks:
            long = block(long, valid)
        fft = self.local_fft(patches, inside)
        frequency_valid = ((inside.float() * self.local_fft.window).gt(0).sum(-1) >= 2) & valid
        eligible = torch.stack((valid, valid, valid, frequency_valid), -1)
        frequency = self.frequency_projection(fft) + code
        evidence = torch.stack((state, short, long, frequency), 2) * eligible[..., None]

        # BTA branch-axis weights are computed from pre-transform representations.
        scores = torch.cat([head(evidence[:, :, r]) for r, head in enumerate(self.route_heads)], -1)
        uniform_branch = eligible.float() / eligible.sum(-1, keepdim=True).clamp_min(1)
        branch_soft = masked_softmax(scores / c.router_temperature, eligible, -1)
        branch_weight = (1 - c.route_floor) * branch_soft + c.route_floor * uniform_branch

        # BST: each source always uses its own transform. No expert reassignment.
        transformed = []
        for r in range(4):
            values = torch.zeros_like(evidence[:, :, r])
            active = eligible[:, :, r]
            values[active] = self.experts[r](evidence[:, :, r][active])
            transformed.append(values)
        features = torch.stack(transformed, 2)
        local_logits = self._classify(features).float() * eligible[..., None]

        # BTA temporal-axis weights are normalized within each branch.
        uniform_patch = eligible.float() / eligible.sum(1, keepdim=True).clamp_min(1)
        pool_scores = torch.cat([head(evidence[:, :, r]) for r, head in enumerate(self.pool_heads)], -1)
        temporal_weight = ((1 - c.pooling_mix) * uniform_patch
                           + c.pooling_mix * masked_softmax(pool_scores, eligible, 1))
        # Keep the original multiplication order, including availability masking.
        contributions = (temporal_weight[..., None] * branch_weight[..., None]
                         * eligible.float()[..., None] * local_logits)
        logits = contributions.sum((1, 2))
        if not return_aux:
            return logits
        return logits, {
            "branch_representations": evidence,
            "branch_available": eligible,
            "branch_weights": branch_weight,
            "temporal_weights": temporal_weight,
            "local_logits": local_logits,
            "signed_contributions": contributions,
            "branch_logits": (uniform_patch[..., None] * local_logits).sum(1),
            "branch_valid": eligible.any(1),
            "patch_position": position,
        }

    def loss(self, logits, labels, aux):
        """Sequence CE plus linearly annealed auxiliary branch CE."""
        c = self.config
        labels = labels.long()
        ce = F.cross_entropy(logits, labels)
        branch_ce = F.cross_entropy(
            aux["branch_logits"].flatten(0, 1),
            labels[:, None].expand(-1, 4).reshape(-1), reduction="none").reshape(-1, 4)
        branch = (branch_ce * aux["branch_valid"]).sum() / aux["branch_valid"].sum().clamp_min(1)
        weight = c.branch_loss_start + self.training_progress * (c.branch_loss_end - c.branch_loss_start)
        return {
            "total": ce + weight * branch, "ce": ce, "branch": branch,
            "branch_weight": logits.new_tensor(weight),
        }
