import os.path as osp
from typing import Optional, Union, Dict, Set

import torch
from torch.utils.data import Dataset
from torch.nn import Module
from tqdm import tqdm

import warnings

from ..utils.helpers import fix_seeds

warnings.filterwarnings("ignore", category=DeprecationWarning)
from tensorboardX import SummaryWriter

from ..configs.train_config import TrainConfig
from ..loggers.log_accumulator import LogAccumulator
from ..utils.storage import Storage
from ..utils import timer


class BaseScheduler:
    def schedule(self, global_step, epoch):
        raise NotImplementedError()


class BaseContext:
    name: str
    trainer: "BaseTrainer"
    model: Module
    dataloader: torch.utils.data.DataLoader
    log_comp: LogAccumulator
    optimizer: Optional[torch.optim.Optimizer]
    writer: SummaryWriter

    def __init__(self, name, trainer, model, dataloader, log_comp, optimizer=None, writer=None):
        self.name = name
        self.trainer = trainer
        self.model = model
        self.dataloader = dataloader
        self.log_comp = log_comp
        self.optimizer = optimizer
        self.writer = writer
        self.run_every = None  # if none uses default values from trainer
        self.num_batches = None  # number of batches to stop evaluation after

        self.local_step = 0
        """ Intra-epoch step counter """

    def compute_loss(self, input: dict):
        raise NotImplementedError


class BaseTrainer:
    """ Object implementing training process """
    model: Module
    datasets: Dict[str, Dataset]
    cfg: TrainConfig
    global_step: int

    Context = BaseContext

    AFTER_BACKWARD = 'after_backward'

    def __init__(self, model: Module,
                 datasets: Union[Dataset, Dict[str, Dataset]],
                 cfg: TrainConfig,
                 storage: Optional[Storage] = None,
                 trainer_state=None,
                 scheduler: Optional[BaseScheduler]=None):
        self.model = model

        if isinstance(datasets, Dataset):
           datasets = {'training': datasets}

        self.datasets = datasets

        self.storage = storage

        self.cfg = cfg

        self.optimizer = self.build_optimizer()

        self.contexts = self.build_contexts()
        """ First context is training, rest are used for validation """

        self.scheduler = scheduler

        self.callbacks = {}

        if trainer_state is not None:
            self.global_step = trainer_state['global_step']
            self.epoch = trainer_state['epoch']
        else:
            self.global_step = 0
            self.epoch = 0

    def save_state(self):
        epoch = self.epoch + 1  # its called after epoch end, so we should start from the next one after restart
        self.storage.save_state(self.model, {'global_step': self.global_step, 'epoch': epoch}, f'epoch_{epoch}')

    def set_after_backward_callback(self, f):
        self.callbacks[self.AFTER_BACKWARD] = f

    @property
    def validation_context_names(self):
        for k in self.datasets:
            if k != 'training':
                yield k

    def build_contexts(self) -> Dict[str, BaseContext]:
        TRAINING = 'training'
        contexts = {}

        assert TRAINING in self.datasets, 'Training is impossible without a dataset'

        writer = SummaryWriter(osp.join(self.storage.experiment_dir, 'tb_training'), flush_secs=3) if self.storage else None
        contexts[TRAINING] = self.Context(TRAINING,
                                          trainer=self,
                                          model=self.model,
                                          optimizer=self.build_optimizer(),
                                          dataloader=self.build_dataloader(TRAINING),
                                          log_comp=LogAccumulator(writer, period=self.cfg.log_period),
                                          writer=writer)

        if self.cfg.val_period == 0:
            return contexts

        for ctx_name in self.validation_context_names:
            val_writer = SummaryWriter(osp.join(self.storage.experiment_dir, 'tb_' + ctx_name), flush_secs=3) if self.storage else None
            contexts[ctx_name] = self.Context(ctx_name,
                                              trainer=self,
                                              model=self.model,
                                              dataloader=self.build_dataloader(ctx_name),
                                              log_comp=LogAccumulator(val_writer, period=0))

        return contexts

    def build_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.learning_rate)

    def build_dataloader(self, ctx_name: str):
        cfg = self.cfg

        def rnd_init(w_id):
            fix_seeds(self.global_step * 100 + w_id)

        return torch.utils.data.DataLoader(self.datasets[ctx_name], batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=rnd_init)

    def train(self, num_steps=None):
        steps = 0

        for self.epoch in range(self.epoch, self.cfg.num_epochs):
            self.model.train()
            ctx = self.contexts['training']
            if ctx.log_comp:
                ctx.log_comp.clear()

            if self.scheduler is not None:
                self.scheduler.step(global_step=self.global_step, epoch=self.epoch)

            for lstep, batch in enumerate(tqdm(ctx.dataloader, total=self.cfg.epoch_size, desc=f'Epoch {self.epoch}')):
                ctx.local_step = lstep
                ctx.log_comp.step = self.global_step

                batch = self.move_to_device(batch, ctx.model.device)
                with timer.measure("compute_loss"):
                    loss = ctx.compute_loss(batch)

                ctx.optimizer.zero_grad()
                with timer.measure('backward'):
                    loss.backward()

                f = self.callbacks.get(self.AFTER_BACKWARD)
                if f is not None:
                    f(ctx)

                ctx.optimizer.step()

                self.global_step += 1
                steps += 1
                if ctx.log_comp:
                    ctx.log_comp.step += 1

                if num_steps is not None and steps == num_steps:
                    print('Target num_steps reached')
                    return

                if lstep == self.cfg.epoch_size - 1:
                    break

            print('Epoch aggregates:')
            if ctx.log_comp:
                ctx.log_comp.print_aggregates()

            self.validation()

            if self.storage:
                self.save_state()

        print('Target epoch number reached')

    def validation(self):
        with torch.no_grad():
            for ctx_name in self.validation_context_names:
                ctx = self.contexts[ctx_name]
                val_period = ctx.run_every or self.cfg.val_period
                if val_period is not None and self.epoch % val_period != 0:
                    continue

                self.model.eval()
                ctx.log_comp.clear()

                for idx, batch in enumerate(ctx.dataloader):
                    ctx.local_step = idx
                    if idx == ctx.num_batches:
                        break
                    batch = self.move_to_device(batch, ctx.model.device)
                    self.process_val_batch(ctx, batch)

                ctx.log_comp.step = self.global_step
                print(f'** Validation results for {ctx_name}:')
                ctx.log_comp.print_aggregates()
                ctx.log_comp.log_aggregates()

    def process_val_batch(self, ctx: BaseContext, batch: dict):
        raise NotImplementedError()

    @property
    def cuda_fields(self) -> Optional[Set[str]]:
        """ Batch fields which should be moved to GPU. If None than all tensors are moved """
        return None

    @timer.wrap('move_to_device')
    def move_to_device(self, batch: Union[dict, torch.Tensor], device):
        res = {}
        if isinstance(batch, torch.Tensor):
            return batch.to(device)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and (self.cuda_fields is None or k in self.cuda_fields):
                v = v.to(device, non_blocking=True)
            res[k] = v
        return res

