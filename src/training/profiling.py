"""Short, isolated training benchmark; never writes or resumes experiment runs."""

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from src.progress import ConsoleProgress


class StageProfiler:
    """CUDA stream intervals, including host launch gaps; not kernel-only time."""

    STAGES = ('transfer', 'forward_loss', 'backward', 'optimizer', 'ema', 'bookkeeping')

    def __init__(self, device, warmup=8):
        self.device = torch.device(device)
        self.warmup = warmup
        self.records = []
        self.current = None
        self.started = None

    def begin(self, step):
        self.current = None
        if step < self.warmup:
            return
        if self.started is None:
            torch.cuda.synchronize(self.device)
            self.started = time.perf_counter()
        self.current = []
        self.mark()

    def mark(self):
        if self.current is not None:
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            self.current.append(event)

    def end(self, updated):
        if self.current is not None:
            self.records.append((self.current, updated))
            self.current = None

    def report(self):
        torch.cuda.synchronize(self.device)
        wall = time.perf_counter() - self.started if self.started is not None else 0
        rows = []
        for events, updated in self.records:
            if len(events) != len(self.STAGES) + 1:
                raise ValueError('Incomplete profiling stage sequence')
            rows.append({**{name: events[i].elapsed_time(events[i + 1])
                            for i, name in enumerate(self.STAGES)}, 'updated': updated})
        count = len(rows)
        updates = sum(row['updated'] for row in rows)
        return {
            'batches': count, 'optimizer_updates': updates,
            'wall_ms_per_batch': wall * 1000 / max(count, 1),
            'cuda_ms_per_batch': {name: sum(row[name] for row in rows) / max(count, 1)
                                  for name in self.STAGES},
            'cuda_ms_per_update': {name: sum(row[name] for row in rows if row['updated']) / max(updates, 1)
                                   for name in ('optimizer', 'ema')},
        }


class BenchmarkBatches:
    def __init__(self, source, count, *, repeat=False):
        self.source, self.count, self.repeat = source, count, repeat

    def __len__(self):
        return self.count

    def __iter__(self):
        if self.repeat:
            for _ in range(self.count):
                yield self.source
        else:
            iterator = iter(self.source)
            for _ in range(self.count):
                yield next(iterator)


def benchmark(config, *, batches=32, warmup=8):
    from src.training.base import set_random_seed
    from src.training.builders import (build_datasets, build_loaders, build_model,
                                       build_optimizer, build_scheduler, build_ema)
    from src.training.engine import ExperimentRunner, train_one_epoch, _move_batch_to_device

    if batches <= 0 or warmup < 0:
        raise ValueError('batches must be positive and warmup nonnegative')
    runner = ExperimentRunner(config)
    if runner.device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError('This benchmark requires CUDA')
    set_random_seed(config.seed)
    train_rows, val_rows = runner._split_data()
    datasets = build_datasets(config, runner.data_workspace, train_rows, val_rows)
    loader, val_loader = build_loaders(config.train, *datasets)
    total = warmup + batches
    if total > len(loader):
        raise ValueError('Benchmark exceeds available training batches')
    # Keep one CPU batch for the isolated replay after workers have been released.
    frozen = next(iter(loader))
    results = {}
    for mode in ('loader', 'gpu_replay'):
        if mode == 'gpu_replay':
            del loader, val_loader, datasets
            gc.collect()
            torch.cuda.empty_cache()
            frozen = _move_batch_to_device(frozen, runner.device)
            frozen['image'] = frozen['image'].contiguous(memory_format=torch.channels_last)
            source = BenchmarkBatches(frozen, total, repeat=True)
        else:
            source = BenchmarkBatches(loader, total)
        set_random_seed(config.seed)
        # Fresh disposable model/optimizer per phase, no pretrained downloads or checkpoints.
        model = build_model(config.model, pretrained=False).to(runner.device, memory_format=torch.channels_last)
        optimizer = build_optimizer(config.train, model)
        ema = build_ema(config.train, model)
        profiler = StageProfiler(runner.device, warmup)
        ConsoleProgress.info(f'Profile {mode}: warmup={warmup}, measured={batches}')
        train_one_epoch(model=model, loader=source, optimizer=optimizer,
                        scheduler=build_scheduler(config.train, optimizer, total),
                        scaler=runner.amp.scaler(), ema=ema, amp=runner.amp,
                        config=config, device=runner.device, profiler=profiler)
        results[mode] = profiler.report()
        ConsoleProgress.info(json.dumps(results[mode], ensure_ascii=False))
        del source, model, optimizer, ema
    return {'gpu': torch.cuda.get_device_name(runner.device), 'config': config.to_dict(),
            'warmup': warmup, 'results': results,
            'note': 'CUDA stream intervals include host launch gaps. Replay uses one fixed batch; no run is saved.'}


def main():
    from src.config import load_experiment_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/local.yaml')
    parser.add_argument('--batches', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--output', default='profiles/local.json')
    args = parser.parse_args()
    report = benchmark(load_experiment_config(args.config), batches=args.batches, warmup=args.warmup)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    ConsoleProgress.info(f'Profile saved: {path}')


if __name__ == '__main__':
    main()
