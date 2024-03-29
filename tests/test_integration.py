import os.path as osp
import shutil
import tempfile
import unittest

import torch

from picograd.configs.train_config import TrainConfig
from picograd.trainers.base import BaseTrainer, BaseContext
from tests.models.linear_net import LinearNet


class RndDataset(torch.utils.data.IterableDataset):
    def __next__(self):
        x = torch.randn([2])
        y = torch.sign(x[0] * x[1])
        return {'x': x, 'y': y}

    def __iter__(self):
        return self


class TstContext(BaseContext):
    def compute_loss(self, input: dict) -> torch.Tensor:
        out = self.model(input['x'])
        loss = torch.nn.functional.nll_loss(out, torch.eq(input['y'], 1).to(out).long())
        self.log_comp.log_metric('loss', loss)
        print('gstep', self.trainer.global_step, 'loss', loss, self.log_comp.running_means['loss'])
        return loss


class TstTrainer(BaseTrainer):
    Context = TstContext


class TstConfig(TrainConfig):
    def build_model(self) -> torch.nn.Module:
        return LinearNet(inp_chans=2, out_chans=2)

    def build_trainer(self, model, storage) -> BaseTrainer:
        ds = RndDataset()
        return TstTrainer(model, ds, cfg=self, storage=storage)


class TestIntegration(unittest.TestCase):
    def setUp(self) -> None:
        self.exps_dir = tempfile.mktemp('test_experiments')

    def tearDown(self) -> None:
        if osp.exists(self.exps_dir):
            shutil.rmtree(self.exps_dir)

    def test_run_epoch(self):
        torch.manual_seed(0)
        cfg = TstConfig()
        cfg.override_attrs(experiment_name='integration_test_exp', num_epochs=3, epoch_size=10, num_workers=0, learning_rate=1)

        trainer = cfg.prepare_trainer(root_dir=self.exps_dir)
        trainer.contexts['training'].log_comp.alpha = 0.1
        trainer.train()
        assert trainer.global_step == 30
        assert trainer.epoch == 3
        assert abs(trainer.contexts['training'].log_comp.running_means['loss'] - -25.42) < 1e-1

        # test loading last checkpoint
        torch.manual_seed(42)
        cfg2 = TstConfig()
        cfg2.override_attrs(experiment_name='integration_test_exp', num_epochs=4, epoch_size=3, num_workers=0, learning_rate=1, device='cuda')
        trainer2 = cfg2.prepare_trainer(root_dir=self.exps_dir, checkpoint='last')
        trainer2.contexts['training'].log_comp.alpha = 0.1
        assert trainer2.global_step == 30
        assert trainer2.epoch == 3
        trainer2.train()

        # test loading named checkpoint

        torch.manual_seed(42)
        cfg3 = TstConfig()
        cfg3.override_attrs(experiment_name='integration_test_exp', num_epochs=4, epoch_size=3, num_workers=0, learning_rate=1, device='cuda')
        trainer3 = cfg3.prepare_trainer(root_dir=self.exps_dir, checkpoint='epoch_3')
        trainer3.contexts['training'].log_comp.alpha = 0.1
        assert trainer3.global_step == 30
        assert trainer3.epoch == 3
        trainer3.train()

        assert trainer3.global_step == 33
        assert trainer3.epoch == 4

        # test determinism

    def test_determinism(self):
        torch.manual_seed(0)
        cfg = TstConfig()
        cfg.override_attrs(experiment_name='integration_test_exp', num_epochs=3, epoch_size=10, num_workers=0, learning_rate=1)

        trainer = cfg.prepare_trainer(root_dir=self.exps_dir)
        trainer.train()
        assert trainer.global_step == 30
        assert trainer.epoch == 3

        cfg2 = TstConfig()
        cfg2.override_attrs(experiment_name='integration_test_exp', num_epochs=3, epoch_size=10, num_workers=0, learning_rate=1)

        trainer2 = cfg.prepare_trainer(root_dir=self.exps_dir, checkpoint='epoch_1')
        trainer2.train()

        assert torch.all(torch.isclose(trainer.model.layer.weight, trainer2.model.layer.weight))
        print('dist', torch.norm(trainer2.model.layer.weight -  trainer2.model.layer.weight))
        assert abs(trainer.contexts['training'].log_comp.running_means['loss'] - trainer2.contexts['training'].log_comp.running_means['loss']) < 1e-8
