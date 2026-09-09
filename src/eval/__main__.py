"""Explicit final holdout evaluation; no threshold search or training here."""

import argparse
import json

from src.eval.checkpoints import CheckpointEvaluator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    evaluator = CheckpointEvaluator(args.run, device=args.device, batch_size=args.batch_size, workers=args.workers)
    print(json.dumps(evaluator.holdout(), indent=2))


if __name__ == '__main__':
    main()
