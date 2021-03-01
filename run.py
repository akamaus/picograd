#!/usr/bin/env python3

from argparse import ArgumentParser
import importlib.util
from pathlib import Path

from picograd.configs.train_config import TrainConfig, BaseConfig

import picograd.utils.timer as timer
from picograd.utils.helpers import fix_seeds

from picograd.utils.config_utils import load_config


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', help='config file to load')
    parser.add_argument('--restore', help='checkpoint name or path')
    parser.add_argument('--fresh_trainer', action='store_true', help='start from first epoch')
    parser.add_argument('--global_step', type=int, help='set starting global_step for trainer')
    parser.add_argument('--train_steps', type=int, help='train for some number of steps and then exit')

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
            trainer.train(args.train_steps)
    finally:
        timer.print_report()
