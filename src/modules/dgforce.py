"""Layer-level DG-Force building blocks for the PVT experiment."""

import torch
from torch import nn
from torch.nn import functional as F


def _bottleneck_width(channels: int, reduction: int) -> int:
    if type(reduction) is not int or reduction < 1:
        raise ValueError('reduction must be a positive integer')
    width = channels // reduction
    if width < 1:
        raise ValueError(f'reduction {reduction} leaves an empty bottleneck for {channels} channels')
    return width


class PatchForensicDisentangle(nn.Module):
    """Per-position MLP bottleneck with no spatial mixing."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        width = _bottleneck_width(channels, reduction)
        self.down = nn.Sequential(nn.Conv2d(channels, width, 1), nn.GELU())
        self.up = nn.Sequential(nn.Conv2d(width, channels, 1), nn.GELU())

    def forward(self, x):
        compressed = self.down(x)
        return self.up(compressed), compressed


class EdgeForensicDisentangle(nn.Module):
    """Spatial convolution bottleneck for boundary-sensitive cues."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        width = _bottleneck_width(channels, reduction)
        self.down = nn.Sequential(
            nn.Conv2d(channels, width, 3, padding=1),
            nn.BatchNorm2d(width),
            nn.GELU(),
        )
        # Neighbourhood mixing happens in the extractor; restoration is a
        # channel projection, which halves the branch cost at full resolution.
        self.up = nn.Sequential(
            nn.Conv2d(width, channels, 1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        compressed = self.down(x)
        return self.up(compressed), compressed


class DFDGLevel(nn.Module):
    """Disentangle patch/edge cues and gather them with token-wise weights."""

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        width = _bottleneck_width(channels, reduction)
        self.patch = PatchForensicDisentangle(channels, reduction)
        self.edge = EdgeForensicDisentangle(channels, reduction)
        self.patch_head = nn.Conv2d(width, 1, 1)
        self.edge_head = nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU(),
                                       nn.Conv2d(width, 1, 1))
        self.evaluator = nn.Sequential(
            nn.Conv2d(channels, width, 1), nn.GELU(), nn.Conv2d(width, 2, 1))
        # Zero gate: every insertion starts as identity, so the pretrained
        # backbone is intact until the cues earn their weight. Without it the
        # thirteen random additions leave stage 4 almost uncorrelated with the
        # pretrained features (cosine ~0.1) and the first epoch trains from scratch.
        self.channel_gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def disentangle(self, x, *, supervise=True):
        patch, patch_compressed = self.patch(x)
        edge, edge_compressed = self.edge(x)
        logits = ((self.patch_head(patch_compressed), self.edge_head(edge_compressed))
                  if supervise else (None, None))
        return patch, edge, *logits

    def forward(self, x, patch_transfer=None, edge_transfer=None, *, supervise=True):
        patch, edge, patch_logits, edge_logits = self.disentangle(x, supervise=supervise)
        if patch_transfer is not None:
            patch = patch_transfer(patch)
        if edge_transfer is not None:
            edge = edge_transfer(edge)
        weights = self.evaluator(x).softmax(dim=1)
        residual = weights[:, :1] * patch + weights[:, 1:] * edge
        enriched = x + self.channel_gate * residual
        return enriched, patch, edge, patch_logits, edge_logits, weights


class IntraScaleTransfer(nn.Module):
    """Gated residual transfer between shallow and deep cues at one scale.

    Both cues are projected into a C/r bottleneck before the 3x3 fusion, so the
    transfer costs a small fraction of a dense 3x3 convolution on C channels.
    """

    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.width = _bottleneck_width(channels, reduction)
        self.project = nn.Sequential(nn.Conv2d(channels, self.width, 1), nn.GELU())
        self.reduce = nn.Conv2d(channels, self.width, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * self.width, self.width, 3, padding=1), nn.GELU(),
            nn.Conv2d(self.width, channels, 1))
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, current, shallow):
        if current.shape != shallow.shape:
            raise ValueError('intra-scale cues must have identical shapes')
        update = self.fuse(torch.cat((self.project(shallow), self.reduce(current)), dim=1))
        return current + self.gate * update


class CrossScaleTransfer(nn.Module):
    """Fine-to-coarse cross-attention followed by a gated residual.

    Queries keep the target resolution; keys and values come from the source
    pooled to the target grid divided by ``kv_stride``, like the spatial
    reduction inside PVT attention. Dense keys would make stage 2 quadratic in
    its 6400 tokens.
    """

    def __init__(self, target_channels: int, source_channels: int, width: int, heads: int,
                 kv_stride: int):
        super().__init__()
        if type(width) is not int or width < 1 or type(heads) is not int or heads < 1:
            raise ValueError('attention width and heads must be positive integers')
        if width % heads:
            raise ValueError('attention width must be divisible by the head count')
        if type(kv_stride) is not int or kv_stride < 1:
            raise ValueError('kv_stride must be a positive integer')
        self.kv_stride = kv_stride
        self.query = nn.Conv2d(target_channels, width, 1)
        self.source = nn.Sequential(nn.Conv2d(source_channels, width, 3, padding=1), nn.GELU())
        self.query_norm = nn.LayerNorm(width)
        self.source_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.output = nn.Conv2d(width, target_channels, 1)
        self.gate = nn.Parameter(torch.zeros(1, target_channels, 1, 1))

    def forward(self, target, source):
        batch, _, height, width = target.shape
        source = F.adaptive_avg_pool2d(
            source, (-(-height // self.kv_stride), -(-width // self.kv_stride)))
        query = self.query_norm(self.query(target).flatten(2).transpose(1, 2))
        context = self.source_norm(self.source(source).flatten(2).transpose(1, 2))
        update, _ = self.attention(query, context, context, need_weights=False)
        update = update.transpose(1, 2).reshape(batch, -1, height, width)
        return target + self.gate * self.output(update)
