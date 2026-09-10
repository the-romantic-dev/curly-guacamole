"""CAT-Net artifact stem followed by our lightweight forensic pyramid."""

import logging

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.dws_conv2d import DWSConv2d


class JPEGArtifactModule(nn.Module):
    """Frequency-wise CAT-Net stem; input is min(abs(quantized Y DCT), 20)."""

    def __init__(self):
        super().__init__()
        self.dc_layer0_dil = nn.Sequential(
            nn.Conv2d(21, 64, 3, dilation=8, padding=8),
            nn.BatchNorm2d(64, momentum=0.01), nn.ReLU(inplace=True))
        self.dc_layer1_tail = nn.Sequential(
            nn.Conv2d(64, 4, 1, bias=False),
            nn.BatchNorm2d(4, momentum=0.01), nn.ReLU(inplace=True))

    def forward(self, bins, qtable):
        categories = torch.arange(21, device=bins.device, dtype=torch.uint8)[None, :, None, None]
        volume = (bins[:, None] == categories).to(self.dc_layer0_dil[0].weight.dtype)
        x = self.dc_layer1_tail(self.dc_layer0_dil(volume))
        b, c, h, w = x.shape
        frequencies = x.reshape(b, c, h//8, 8, w//8, 8).permute(0, 1, 3, 5, 2, 4)
        raw = frequencies.reshape(b, c*64, h//8, w//8)
        weighted = (frequencies * qtable.to(x.dtype)[:, None, :, :, None, None]).reshape_as(raw)
        return torch.cat((raw, weighted), dim=1)

    def load_pretrained(self, path):
        """Load the complete CAT-Net artifact stem; ignore its HRNet and heads."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        state = checkpoint.get('state_dict', checkpoint)
        state = {key.removeprefix('module.'): value for key, value in state.items()}
        expected = self.state_dict()
        selected = {key: state[key] for key in expected if key in state}
        self.load_state_dict(selected, strict=True)
        logging.getLogger(__name__).info('Loaded JPEG artifact weights from %s (%d tensors)', path, len(selected))


class JPEGBranch(nn.Module):
    """Process each native frame without batch padding; align features for fusion."""

    def __init__(self, channels):
        super().__init__()
        a, b, c = channels
        self.channels_by_stride = dict(zip((8, 16, 32), channels, strict=True))
        self.artifact = JPEGArtifactModule()
        self.project = nn.Sequential(nn.Conv2d(512, a, 1, bias=False), nn.BatchNorm2d(a), nn.GELU())
        self.refine = DWSConv2d(a, a)
        self.down16 = DWSConv2d(a, b, stride=2)
        self.down32 = DWSConv2d(b, c, stride=2)

    @staticmethod
    def align(feature, geometry, stride, target_size, *, orientation=1, source_size=None):
        top, left, height, width, rotations, horizontal, vertical = geometry
        # Crop/scale in pixel coordinates: partial JPEG blocks must not stretch
        # their padded area across the valid image. Rotation is applied afterwards.
        out_h, out_w = target_size[::-1] if rotations % 2 else target_size
        y = top + (torch.arange(out_h, device=feature.device, dtype=torch.float32) + .5) * height / out_h
        x = left + (torch.arange(out_w, device=feature.device, dtype=torch.float32) + .5) * width / out_w
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        if orientation != 1:
            h, w = source_size
            # Inverse EXIF mapping, from displayed image to stored JPEG coordinates.
            xx, yy = {2: (w-xx, yy), 3: (w-xx, h-yy), 4: (xx, h-yy),
                      5: (yy, xx), 6: (yy, h-xx), 7: (w-yy, h-xx), 8: (w-yy, xx)}[orientation]
        yy = 2*yy/(stride*feature.shape[-2])-1
        xx = 2*xx/(stride*feature.shape[-1])-1
        grid = torch.stack((xx, yy), dim=-1)[None]
        aligned = F.grid_sample(feature.float(), grid, padding_mode='border', align_corners=False).to(feature.dtype)
        aligned = torch.rot90(aligned, rotations, dims=(-2, -1))
        if horizontal:
            aligned = aligned.flip(-1)
        if vertical:
            aligned = aligned.flip(-2)
        return aligned

    def forward(self, inputs, target_sizes):
        outputs = {stride: [] for stride in self.channels_by_stride}
        for sample in inputs:
            x = self.refine(self.project(self.artifact(sample['bins'][None], sample['qtable'][None])))
            x16 = self.down16(x)
            features = {8: x, 16: x16, 32: self.down32(x16)}
            for stride, feature in features.items():
                outputs[stride].append(self.align(feature, sample['geometry'], stride, target_sizes[stride],
                                                   orientation=sample.get('orientation', 1),
                                                   source_size=sample.get('source_size')))
        return {stride: torch.cat(features) for stride, features in outputs.items()}
