"""CPU-only microbenchmark of local preprocessing; no dataset or weights needed."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from src.data.local_preprocess import LocalPreprocessor


class PreviousPreprocessor(LocalPreprocessor):
    """Reference: full float32 CHW buffer followed by one full-tensor cast."""

    @staticmethod
    def _colors(image):
        rgb = image.astype(np.float32) / 255.
        return rgb, cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)

    @staticmethod
    def _normalize(rgb):
        return (rgb - np.asarray(IMAGENET_DEFAULT_MEAN, np.float32)) / np.asarray(IMAGENET_DEFAULT_STD, np.float32)

    def __call__(self, image, *, dtype=torch.float32):
        rgb, ycrcb = self._colors(image)
        rgb = self._normalize(self._resize(rgb))
        features = np.empty((15, self.image_size, self.image_size), np.float32)
        features[:3] = rgb.transpose(2, 0, 1)
        residual = np.empty_like(ycrcb)
        for axis, signed, absolute in ((1, 3, 9), (0, 6, 12)):
            self._difference(ycrcb, residual, axis=axis)
            features[signed:signed + 3] = self._resize(residual).transpose(2, 0, 1)
            np.abs(residual, out=residual)
            features[absolute:absolute + 3] = self._resize(residual).transpose(2, 0, 1)
        return torch.from_numpy(features).to(dtype)


class TimedPreprocessor(LocalPreprocessor):
    def __call__(self, image, **kwargs):
        self.timings = defaultdict(float)
        return super().__call__(image, **kwargs)

    def _measure(self, name, function, *args, **kwargs):
        started = time.perf_counter()
        result = function(*args, **kwargs)
        self.timings[name] += (time.perf_counter() - started) * 1000
        return result

    def _colors(self, image):
        return self._measure('colors', super()._colors, image)

    def _normalize(self, image):
        return self._measure('normalize', super()._normalize, image)

    def _difference(self, *args, **kwargs):
        return self._measure('differences', super()._difference, *args, **kwargs)

    def _resize(self, image):
        return self._measure('resize', super()._resize, image)

    def _write(self, features, start, values):
        started = time.perf_counter()
        converted = torch.from_numpy(values).to(features.dtype)
        self.timings['cast'] += (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        features[start:start + 3].copy_(converted.permute(2, 0, 1))
        self.timings['write_chw'] += (time.perf_counter() - started) * 1000


def measure(shape, *, size=1024, dtype=torch.bfloat16, repeats=9):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    image = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    previous, current = PreviousPreprocessor(size), TimedPreprocessor(size)
    torch.testing.assert_close(previous(image, dtype=dtype), current(image, dtype=dtype), rtol=0, atol=0)
    durations = {'previous': [], 'current': []}
    stages = defaultdict(list)
    for iteration in range(repeats):
        order = [('previous', previous), ('current', current)]
        for name, processor in (order if iteration % 2 == 0 else reversed(order)):
            started = time.perf_counter()
            output = processor(image, dtype=dtype)
            elapsed = (time.perf_counter() - started) * 1000
            durations[name].append(elapsed)
            if name == 'current':
                for stage, duration in processor.timings.items():
                    stages[stage].append(duration)
                stages['other'].append(max(0., elapsed - sum(processor.timings.values())))
            del output
    medians = {name: float(np.median(values)) for name, values in durations.items()}
    return {'input_shape': shape, 'output_size': size, 'dtype': str(dtype),
            'median_ms': medians, 'speedup': medians['previous'] / medians['current'],
            'current_stage_mean_ms': {name: float(np.mean(values)) for name, values in stages.items()},
            'equal': True, 'repeats': repeats}


def main():
    from src.training.builders import DataLoaderThreadLimits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dtype', choices=['float32', 'float16', 'bfloat16'], default='bfloat16')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=9)
    parser.add_argument('--output')
    args = parser.parse_args()
    DataLoaderThreadLimits.apply()
    reports = [measure(shape, size=args.size, dtype=getattr(torch, args.dtype), repeats=args.repeats)
               for shape in ((640, 640), (1024, 1024), (1536, 2048))]
    text = json.dumps(reports, indent=2)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')


if __name__ == '__main__':
    main()
