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

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        rgb = image.astype(np.float32) / 255.0
        ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
        rgb = self._resize(rgb)
        rgb = (rgb - np.asarray(IMAGENET_DEFAULT_MEAN, dtype=np.float32)) / np.asarray(
            IMAGENET_DEFAULT_STD, dtype=np.float32)
        # OpenCV's fractional area path only supports <=4 channels. Reducing
        # each three-plane group also avoids a native-resolution 12-channel copy.
        # Write directly to the final CHW buffer, avoiding concatenation followed
        # by another full 60 MiB layout copy. Reuse one native residual buffer.
        features = np.empty((15, self.image_size, self.image_size), dtype=np.float32)
        features[:3] = rgb.transpose(2, 0, 1)
        residual = np.empty_like(ycrcb)
        np.subtract(ycrcb[:, 1:], ycrcb[:, :-1], out=residual[:, :-1])
        residual[:, -1] = 0
        features[3:6] = self._resize(residual).transpose(2, 0, 1)
        np.abs(residual, out=residual)
        features[9:12] = self._resize(residual).transpose(2, 0, 1)
        np.subtract(ycrcb[1:], ycrcb[:-1], out=residual[:-1])
        residual[-1] = 0
        features[6:9] = self._resize(residual).transpose(2, 0, 1)
        np.abs(residual, out=residual)
        features[12:15] = self._resize(residual).transpose(2, 0, 1)
        return torch.from_numpy(features)

    def _resize(self, image):
        # Resize each dimension separately when stretch shrinks one and grows the other.
        h, w = image.shape[:2]
        size = self.image_size
        if w != size:
            image = cv2.resize(image, (size, h), interpolation=cv2.INTER_AREA if w > size else cv2.INTER_LINEAR)
        if h != size:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA if h > size else cv2.INTER_LINEAR)
        return image
