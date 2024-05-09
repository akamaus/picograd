import importlib.util
from pathlib import Path
from picograd.configs.train_config import TrainConfig, BaseConfig


def load_config(config_path: str, rest_args: list) -> TrainConfig:
    if not Path(config_path).exists():
        raise ValueError("Config not found", config_path)
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
