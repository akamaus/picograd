import unittest
import pytest
import shutil
import tempfile

import torch

from picograd.utils.storage import Storage
from tests.models.linear_net import LinearNet


class TestUtils(unittest.TestCase):
    def setUp(self) -> None:
        self.exps_dir = tempfile.mktemp('test_experiments')

    def tearDown(self) -> None:
        shutil.rmtree(self.exps_dir)

    def test_save_load(self):
        model = LinearNet(inp_chans=2, out_chans=3)
        storage = Storage(self.exps_dir, 'exp1')

        state = {'epoch': 12}
        storage.save_state(model, state, cpt_name='epoch1')

        cpt_path = storage.checkpoint_path('epoch1')
        model2, state2 = storage.load_state(checkpoint_path=cpt_path)
        assert torch.all(torch.eq(model.layer.weight.cpu(), model2.layer.weight.cpu()))
        assert state2 == state

        model3, state3 = storage.load_state(checkpoint_name='epoch1')
        assert torch.all(torch.eq(model.layer.weight.cpu(), model3.layer.weight.cpu()))
        assert state3 == state

    def test_save_load_last(self):
        model = LinearNet(inp_chans=2, out_chans=3)
        storage = Storage(self.exps_dir, 'exp1')

        state = {'epoch': 1}
        storage.save_state(model, state, cpt_name='epoch1')

        state = {'epoch': 2}
        storage.save_state(model, state, cpt_name='epoch2')

        storage2 = Storage(self.exps_dir, 'exp1')
        model2, state2 = storage2.load_state()
        assert torch.all(torch.eq(model.layer.weight.cpu(), model2.layer.weight.cpu()))
        assert state2 == state
