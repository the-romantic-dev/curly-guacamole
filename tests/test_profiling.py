import cv2
import numpy as np
import torch
import pytest

from src.training.profiling import BenchmarkBatches, StageProfiler


def test_benchmark_batches_are_bounded_and_replay_does_not_copy():
    assert list(BenchmarkBatches(range(10), 3)) == [0, 1, 2]
    batch = {'image': torch.zeros(1)}
    repeated = list(BenchmarkBatches(batch, 4, repeat=True))
    assert len(repeated) == 4 and all(item is batch for item in repeated)


def test_profiler_warmup_and_per_update_denominators(monkeypatch):
    class Event:
        counter = 0

        def __init__(self, **kwargs):
            self.index = Event.counter
            Event.counter += 1

        def record(self, stream):
            pass

        def elapsed_time(self, other):
            return (other.index - self.index) * 2.

    monkeypatch.setattr(torch.cuda, 'Event', Event)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: None)
    profiler = StageProfiler('cuda', warmup=1)
    for step in range(3):
        profiler.begin(step)
        for _ in profiler.STAGES:
            profiler.mark()
        profiler.end(step == 2)
    report = profiler.report()
    assert report['batches'] == 2
    assert report['optimizer_updates'] == 1
    assert report['cuda_ms_per_batch']['backward'] == 2.
    assert report['cuda_ms_per_update']['ema'] == 2.


def test_benchmark_rejects_cpu_before_loading_data():
    from dataclasses import replace
    from src.config import load_experiment_config
    from src.training.profiling import benchmark
    cfg = load_experiment_config('configs/local.yaml')
    cfg = replace(cfg, train=replace(cfg.train, device='cpu'))
    with pytest.raises(ValueError, match='CUDA'):
        benchmark(cfg)
