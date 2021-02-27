from typing import Optional
from collections import defaultdict

import torch
from tensorboardX import SummaryWriter


class LogAccumulator:
    """ A simplistic object accumulating the performance statistics """
    def __init__(self, writer: Optional[SummaryWriter], alpha=0.99, period=1):
        assert 0 < alpha < 1
        self.epoch_logs = defaultdict(lambda: [])
        self.running_means = dict()
        self.alpha = alpha
        """ the larger the longer running memory is """
        self.period = period
        """ How often to log """
        self.writer = writer
        self.step = 0

    def state_dict(self):
        return {'epoch_logs': dict(self.epoch_logs),
                'running_means': self.running_means,
                'step': self.step
               }

    def load_state_dict(self, state, strict=True):
        self.epoch_logs.clear()
        for k, log in state['epoch_logs'].items():
            self.epoch_logs[k] = log

        self.running_means = state['running_means']
        self.step = state['step']

    def log_metric(self, name, value, aggregate=True):
        """ store value in internal array and optionally log it to TB """
        if isinstance(value, torch.Tensor):
            value = value.item()

        if aggregate:
            self.epoch_logs[name].append(value)
            if name in self.running_means:
                self.running_means[name] = self.alpha * self.running_means[name] + (1 - self.alpha) * value
            else:
                self.running_means[name] = value

        write = False
        if self.writer is not None:
            if self.period > 1:
                if self.step % self.period == 0:
                    write = True
            elif self.period == 0:
                write = False
            else:
                write = True

        if write:
            self.writer.add_scalar(name, value, self.step)

    def log_aggregates(self):
        """ Log accumulated stats to protobuf (primarily used during validation) """
        if self.writer is None:
            return

        for n, vals in self.epoch_logs.items():
            self.writer.add_scalar(n, torch.tensor(vals).float().mean(), global_step=self.step)

    def log_text(self, label, text):
        """ Log some text """
        self.writer.add_text(label, text, global_step=self.step)

    def print_aggregates(self):
        """ Print aggregated value to stdout """
        for n, vs in self.epoch_logs.items():
            t = torch.tensor(vs).float()
            run = self.running_means[n]
            mean = t.mean()
            std = t.std()
            print(f'{n}: running: {run:.5f}; mean: {mean:.5f} +- {std:.5f}')

    def clear(self):
        self.epoch_logs.clear()
