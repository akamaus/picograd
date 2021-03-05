import os.path as osp
from typing import Optional, Union, Dict, Set, Callable

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
from ..utils.storage import Storage, LoadStateError
from ..utils import timer


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

    def state_dict(self):
        state = {}

        opt = None
        if self.optimizer is not None:
            if isinstance(self.optimizer, dict):
                opt_st = {}
                for k, o in self.optimizer.items():
                    opt_st[k] = o.state_dict()
            else:
                opt_st = self.optimizer.state_dict()

        state['optimizer'] = opt_st
        state['log_comp'] = self.log_comp.state_dict() if self.log_comp else None
        return state

    def load_state_dict(self, state, strict=True):
        opt_st = state['optimizer']

        if (opt_st is None) != (self.optimizer is None):
            if strict:
                raise LoadStateError('no state for optimizer or vise-versa')
        elif self.optimizer is not None:
            if isinstance(self.optimizer, dict):
                for n, opt in self.optimizer.items():
                    opt.load_state_dict(opt_st[n])
            else:
                self.optimizer.load_state_dict(opt_st)

        if self.log_comp is not None:
            self.log_comp.load_state_dict(state['log_comp'])


class BaseTrainer:
    """ Object implementing training process """
    model: Module
    datasets: Dict[str, Dataset]
    cfg: TrainConfig
    global_step: int

    Context = BaseContext

    AFTER_BACKWARD_CALLBACK = 'after_backward'
    AFTER_STEP_CALLBACK = 'after_step'
    AFTER_EPOCH_CALLBACK = 'after_epoch'

    def __init__(self, model: Module,
                 datasets: Union[Dataset, Dict[str, Dataset]],
                 cfg: TrainConfig,
                 storage: Optional[Storage] = None):
        self.model = model

        if isinstance(datasets, Dataset):
           datasets = {'training': datasets}

        self.datasets = datasets

        self.storage = storage

        self.cfg = cfg

        self.contexts = self.build_contexts()
        """ First context is training, rest are used for validation """

        self.callbacks = {}

        self.global_step = 0
        self.epoch = 0

    def save_state(self):
        self.storage.save_state(self.model, self.state_dict(), f'epoch_{self.epoch}')

    def state_dict(self):
        ctxs = {k: ctx.state_dict() for k,ctx in self.contexts.items()}
        state = {'global_step': self.global_step,
                 'epoch': self.epoch,
                 'contexts': ctxs}
        return state

    def load_state_dict(self, state, strict=True):
        self.global_step = state['global_step']
        self.epoch = state['epoch']

        ctxs_st = state.get('contexts')
        if ctxs_st is None:
            if strict:
                raise LoadStateError('no contexts in state')
        else:
            if len(ctxs_st) != len(self.contexts) and strict:
                raise LoadStateError('Excessive contexts in checkpoint', list(ctxs_st.keys()))

            for k, ctx in self.contexts.items():
                st = ctxs_st.get(k)
                if st is None:
                    if strict:
                        raise LoadStateError('no state for context', k)
                else:
                    ctx.load_state_dict(ctxs_st[k])


    def add_callback(self, callback_name:str, f:Callable):
        lst = self.callbacks.get(callback_name, [])
        lst.append(f)
        self.callbacks[callback_name] = lst

    def execute_callbacks(self, callback_name:str, ctx: BaseContext):
        lst = self.callbacks.get(callback_name, [])
        for clbk in lst:
            clbk(ctx)

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

        for ctx_name in self.validation_context_names:
            val_writer = SummaryWriter(osp.join(self.storage.experiment_dir, 'tb_' + ctx_name), flush_secs=3) if self.storage else None
            contexts[ctx_name] = self.Context(ctx_name,
                                              trainer=self,
                                              model=self.model,
                                              dataloader=self.build_dataloader(ctx_name),
                                              log_comp=LogAccumulator(val_writer, period=0))

        return contexts

    def build_optimizer(self):
        def adam_opt(model):
            return torch.optim.Adam(model.parameters(), lr=self.cfg.learning_rate, betas=(self.cfg.beta1, self.cfg.beta2))

        if isinstance(self.model, dict):
            res = {}
            for name, module in self.model.items():
                res[name] = adam_opt(module)
        else:
            res = adam_opt(self.model)

        return res

    def build_dataloader(self, ctx_name: str):
        cfg = self.cfg

        def rnd_init(w_id):
            fix_seeds(self.global_step * 100 + w_id)

        return torch.utils.data.DataLoader(self.datasets[ctx_name], batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=rnd_init)

    def update_model(self, ctx, loss):
        ctx.optimizer.zero_grad()
        with timer.measure('backward'):
            loss.backward()

        self.execute_callbacks(self.AFTER_BACKWARD_CALLBACK, ctx)

        ctx.optimizer.step()

    def train(self, num_steps=None):
        steps = 0
        early_finish = False

        while True:
            if isinstance(self.model, dict):
                for m in self.model.values():
                    m.train()
            else:
                self.model.train()

            ctx = self.contexts['training']
            if ctx.log_comp:
                ctx.log_comp.clear()

            for lstep, batch in enumerate(tqdm(ctx.dataloader, total=self.cfg.epoch_size, desc=f'Epoch {self.epoch}')):
                ctx.local_step = lstep
                ctx.log_comp.step = self.global_step

                device = next(iter(ctx.model.values())).device if isinstance(ctx.model, dict) else ctx.model.device

                batch = self.move_to_device(batch, device)
                with timer.measure("compute_loss"):
                    loss = ctx.compute_loss(batch)

                ctx.optimizer.zero_grad()
                with timer.measure('backward'):
                    loss.backward()

                self.execute_callbacks(self.AFTER_BACKWARD_CALLBACK, ctx)

                ctx.optimizer.step()

                self.execute_callbacks(self.AFTER_STEP_CALLBACK, ctx)

                self.global_step += 1
                steps += 1
                if ctx.log_comp:
                    ctx.log_comp.step += 1

                if num_steps is not None and steps == num_steps:
                    print('Target num_steps reached')
                    early_finish = True
                    break

                if lstep == self.cfg.epoch_size - 1:
                    break

            self.execute_callbacks(self.AFTER_EPOCH_CALLBACK, ctx)

            print('Epoch aggregates:')
            if ctx.log_comp:
                ctx.log_comp.print_aggregates()

            self.validation()

            self.epoch += 1

            if self.storage:
                self.save_state()

            if self.epoch >= self.cfg.num_epochs:
                print('Target epoch number reached')
                break

            if early_finish:
                break

    def validation(self):
        with torch.no_grad():
            for ctx_name in self.validation_context_names:
                ctx = self.contexts[ctx_name]
                val_period = ctx.run_every or self.cfg.val_period
                if val_period is not None or val_period > 0 or self.epoch % val_period != 0:
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
    def move_to_device(self, batch: Union[dict, tuple, torch.Tensor], device):
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        elif isinstance(batch, (tuple, list)):
            res = []
            for k, v in enumerate(batch):
                if isinstance(v, torch.Tensor) and (self.cuda_fields is None or k in self.cuda_fields):
                    v = v.to(device, non_blocking=True)
                res.append(v)
        elif isinstance(batch, dict):
            res = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and (self.cuda_fields is None or k in self.cuda_fields):
                    v = v.to(device, non_blocking=True)
                res[k] = v
        else:
            raise RuntimeError('Strange batch type', type(batch))

        return res

    def process_val_batch(self, ctx: BaseContext, batch: dict):
        ctx.compute_loss(batch)
