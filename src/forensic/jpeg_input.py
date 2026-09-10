"""Native JPEG magnitude bins and geometry, without RGB decoding or DCT resize."""

import io
import os
import sys
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from PIL import Image


@lru_cache(maxsize=1)
def _decoder():
    from cffi import FFI

    ffi = FFI()
    ffi.cdef('''
        typedef struct { unsigned char *bins; int height, width, rows, cols;
            unsigned short qtable[64]; char error[256]; } jpeg_result;
        int read_jpeg_bins(const unsigned char *, unsigned long, jpeg_result *);
        void free_jpeg_bins(jpeg_result *);
    ''')
    prefix = Path(sys.prefix) / ('Library' if os.name == 'nt' else '')
    # Compile once in the parent before workers start; persistent cache is local.
    cache = Path(__file__).resolve().parents[2] / 'runs' / '.jpeg_decoder'
    cache.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        _decoder.dll_directory = os.add_dll_directory(str(prefix / 'bin'))
    library = ffi.verify(Path(__file__).with_name('jpeg_coefficients.c').read_text(),
                         include_dirs=[str(prefix / 'include')],
                         library_dirs=[str(prefix / 'lib')],
                         libraries=['libjpeg' if os.name == 'nt' else 'jpeg'],
                         extra_link_args=['/MANIFEST:NO'] if os.name == 'nt' else [],
                         tmpdir=str(cache))
    return ffi, library


@dataclass(frozen=True)
class JPEGInput:
    bins: np.ndarray
    qtable: np.ndarray
    # Crop in native pixels, then rotation and horizontal/vertical flips.
    geometry: tuple[int, int, int, int, int, int, int]
    orientation: int = 1
    source_size: tuple[int, int] | None = None
    available: bool = True

    @staticmethod
    def prepare_decoder():
        _decoder()

    @classmethod
    def read(cls, source):
        data = source if isinstance(source, bytes) else Path(source).read_bytes()
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            # Some originals are PNG files named .jpg. They have no JPEG DCT.
            with Image.open(io.BytesIO(data)) as image:
                w, h = image.size
                image.verify()
            return cls(np.empty((0, 0), np.uint8), np.ones((8, 8), np.float32),
                       (0, 0, h, w, 0, 0, 0), source_size=(h, w), available=False)
        ffi, decoder = _decoder()
        result = ffi.new('jpeg_result *')
        if not decoder.read_jpeg_bins(data, len(data), result):
            origin = '<in-memory JPEG>' if isinstance(source, bytes) else str(source)
            reason = ffi.string(result.error).decode(errors='replace')
            raise ValueError(f'Cannot read JPEG [{origin}]: {reason}')
        try:
            bins = np.frombuffer(ffi.buffer(result.bins, result.rows * result.cols), np.uint8).copy()
            bins = bins.reshape(result.rows, result.cols)
            qtable = np.frombuffer(ffi.buffer(result.qtable), np.uint16).astype(np.float32).reshape(8, 8)
            with Image.open(io.BytesIO(data)) as image:
                orientation = int(image.getexif().get(274, 1))
            if orientation not in range(1, 9):
                raise ValueError('Unsupported JPEG EXIF orientation')
            h, w = result.height, result.width
            oriented = (w, h) if orientation >= 5 else (h, w)
            return cls(bins, qtable, (0, 0, *oriented, 0, 0, 0), orientation, (h, w))
        finally:
            decoder.free_jpeg_bins(result)

    def crop(self, top, left, height, width):
        return replace(self, geometry=(top, left, height, width, 0, 0, 0))

    def transform(self, rotations, horizontal, vertical):
        return replace(self, geometry=(*self.geometry[:4], rotations, int(horizontal), int(vertical)))

    def tensors(self):
        return {'bins': torch.from_numpy(self.bins), 'qtable': torch.from_numpy(self.qtable),
                'geometry': self.geometry, 'orientation': self.orientation, 'source_size': self.source_size,
                'available': self.available}
