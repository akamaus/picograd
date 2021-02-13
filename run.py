#!/usr/bin/env python3

from argparse import ArgumentParser
import importlib.util
from pathlib import Path

from configs.train_config import TrainConfig, BaseConfig

import utils.timer as timer
from utils.helpers import fix_seeds


def load_config(config_path: str, rest_args: list) -> TrainConfig:
    spec = importlib.util.spec_from_file_location(Path(config_path).stem, config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    arg_tuples = []
    while len(rest_args) > 0:
        arg = rest_args.pop(0)
        value = rest_args.pop(0)
        if not arg.startswith('--'):
            raise ValueError(f"Can't parse additional argument pair ({arg}, {value}")

        arg_tuples.append((arg[2:], value))

    cfg_args = dict(arg_tuples)
    cfg = config.Config()
    cfg.override_attrs(**cfg_args)
    print('Config', cfg)

    assert isinstance(cfg, BaseConfig)
    return cfg


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='config file to load')
    parser.add_argument('--restore', help='checkpoint name or path')
    parser.add_argument('--fresh_trainer', action='store_true', help='start from first epoch')
    parser.add_argument('--global_step', type=int, help='set starting global_step for trainer')

    args, rest_args = parser.parse_known_args()
    cfg = load_config(args.config, rest_args)

    trainer = cfg.prepare_trainer(checkpoint=args.restore, fresh_trainer = cfg.fresh_trainer or args.fresh_trainer)
    if args.global_step:
        trainer.global_step = args.global_step
        trainer.epoch = args.global_step // cfg.epoch_size

    print('Model:')
    print(trainer.model)

    print('Fixing seeds to 42')
    fix_seeds(42)

    try:
        with timer.measure('total_train'):
            trainer.train()
    finally:
        timer.print_report()
