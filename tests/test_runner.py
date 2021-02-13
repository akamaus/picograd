import os.path as osp

import pytest

from run import load_config


def test_runner():
    dir = osp.dirname(__file__)
    cfg_path = osp.join(dir, 'configs', 'config_for_test.py')

    with pytest.raises(ValueError):
        load_config(cfg_path, [])

    cfg = load_config(cfg_path, ['--b', "123"])
    assert cfg.a == 2
    assert cfg.b == "123"
