"""Run one configured experiment, selecting checkpoints on development only."""

import argparse

from src.config import load_experiment_config
from src.training.engine import ExperimentRunner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    ExperimentRunner(load_experiment_config(args.config)).run()


if __name__ == '__main__':
    main()
