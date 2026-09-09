"""Native residual extraction for a synchronized high-resolution local view."""

import cv2
import numpy as np
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class LocalPreprocessor:
    """RGB + signed dx/dy + absolute dx/dy for float Y, Cr, Cb (15 channels).

    Geometry and appearance augmentation must already have been applied. Extract
    differences before spatial reduction; absolute channels retain energy when
    opposite signed responses cancel. The boundary uses replicated pixels.
    """

    CHANNELS = 15

    def __init__(self, image_size: int):
        if type(image_size) is not int or image_size < 32 or image_size % 32:
            raise ValueError('local_image_size must be a positive multiple of 32')
        self.image_size = image_size

    def __call__(self, image: np.ndarray, *, dtype=torch.float32) -> torch.Tensor:
        rgb, ycrcb = self._colors(image)
        rgb = self._normalize(self._resize(rgb))
        # OpenCV's fractional area path only supports <=4 channels. Reducing
        # each three-plane group also avoids a native-resolution 12-channel copy.
        # Allocate only the final output dtype, not a full float32 CHW staging
        # tensor. Reuse one native residual buffer for both derivative axes.
        features = torch.empty((15, self.image_size, self.image_size), dtype=dtype, device='cpu')
        self._write(features, 0, rgb)
        residual = np.empty_like(ycrcb)
        self._difference(ycrcb, residual, axis=1)
        self._write(features, 3, self._resize(residual))
        np.abs(residual, out=residual)
        self._write(features, 9, self._resize(residual))
        self._difference(ycrcb, residual, axis=0)
        self._write(features, 6, self._resize(residual))
        np.abs(residual, out=residual)
        self._write(features, 12, self._resize(residual))
        return features

    @staticmethod
    def _colors(image):
        rgb = image.astype(np.float32)
        rgb /= np.float32(255.)
        return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)

    @staticmethod
    def _normalize(rgb):
        for channel, (mean, std) in enumerate(zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)):
            plane = rgb[..., channel]
            np.subtract(plane, np.float32(mean), out=plane)
            np.divide(plane, np.float32(std), out=plane)
        return rgb

    @staticmethod
    def _difference(ycrcb, residual, *, axis):
        if axis == 1:
            np.subtract(ycrcb[:, 1:], ycrcb[:, :-1], out=residual[:, :-1])
            residual[:, -1] = 0
        else:
            np.subtract(ycrcb[1:], ycrcb[:-1], out=residual[:-1])
            residual[-1] = 0

    @staticmethod
    def _write(features, start, values):
        # Contiguous HWC cast is faster than casting a strided CHW view.
        # Only one three-channel compact temporary, never a full float32 output.
        converted = torch.from_numpy(values).to(features.dtype)
        features[start:start + 3].copy_(converted.permute(2, 0, 1))

    def _resize(self, image):
        # Resize each dimension separately when stretch shrinks one and grows the other.
        h, w = image.shape[:2]
        size = self.image_size
        if w != size:
            image = cv2.resize(image, (size, h), interpolation=cv2.INTER_AREA if w > size else cv2.INTER_LINEAR)
        if h != size:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA if h > size else cv2.INTER_LINEAR)
        return image
