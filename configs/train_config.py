""" Here Config objects specifying details of training process live """
import os.path as osp
import torch
from typing import TYPE_CHECKING, Union, Tuple

from configs.base import BaseConfig
from utils.storage import Storage


if TYPE_CHECKING:
    from trainers.base import BaseTrainer


class TrainConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.experiment_name = self.__class__.__module__.split('.')[-1]
        self.nsuf = ""
        self.num_workers = 2
        self.batch_size = 32
        self.epoch_size = 100
        self.num_epochs = 1000
        self.learning_rate = 1e-4
        self.log_period = 1
        self.val_period = 0  # in epochs
        self.val_batches = 10
        self.fresh_trainer = False

        self.baseline_tau = 0.01
        self.device = 'cpu'
        self.use_meta_info = True
        """ Should we build the model from metainfo in checkpoint or just by calling build_model """

    def load_model(self, root_dir: str = 'experiments', checkpoint: str = 'last', with_storage=False) -> Union[torch.nn.Module, Tuple[torch.nn.Module, Storage, dict]]:
        exp_name = self.experiment_name
        if self.nsuf:
            exp_name += "_" + self.nsuf

        storage = Storage(root_dir=root_dir, experiment_name=exp_name)
        trainer_state = None

        if self.use_meta_info:
            model = self.build_model().__class__
        else:
            model = self.build_model()

        if checkpoint is not None:
            if checkpoint.find(osp.sep) > -1:
                model, trainer_state = storage.load_state(checkpoint_path=checkpoint, model=model)
            else:
                if checkpoint == 'last':
                    checkpoint = None
                model, trainer_state = storage.load_state(checkpoint_name=checkpoint, model=model)

        model = model.to(self.device)
        model.device = self.device
        if with_storage:
            return model, storage, trainer_state
        else:
            return model

    def prepare_trainer(self, root_dir='experiments', checkpoint=None, fresh_trainer=False) -> "BaseTrainer":
        model, storage, train_state = self.load_model(root_dir, checkpoint, with_storage=True)
        if fresh_trainer:
            train_state = None
        return self.build_trainer(model, storage=storage, trainer_state=train_state)

    def build_model(self) -> torch.nn.Module:
        """ Build model class from scratch """
        raise NotImplemented()

    def build_trainer(self, model: torch.nn.Module, storage: Storage, trainer_state: dict) -> "BaseTrainer":
        raise NotImplemented()
