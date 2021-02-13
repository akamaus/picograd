""" Time measuring helpers """
from time import time

from tabulate import tabulate
import torch


__all__ = ["Timer", "measure", "wrap", "print_report"]


class Timer:
    """ Object providing context wrappers and decorators for convenient time measuring """
    def __init__(self, timer_name='default'):
        self._name = timer_name
        self._timers = {}

    def _report(self, name:str, dt:float):
        """ Account single measurement """
        if name not in self._timers:
            self._timers[name] = []
        self._timers[name].append(dt)

    def print_report(self):
        print(f"Time stats for {self._name}:")
        table_head = ['Measure', 'Mean', 'Sum', 'Min', 'Max', 'Std', 'Calls']
        table_rows = []
        for name, vals in self._timers.items():
            vals = torch.tensor(vals)
            line = [name, vals.mean().item(), vals.sum().item(), vals.min().item(), vals.max().item(), vals.std().item(), len(vals)]
            table_rows.append(line)

        print(tabulate(table_rows, headers=table_head, floatfmt=".6f"))


    def measure(self, name):
        return Measure(self, name)

    def wrap(self, name:str):
        def wrapping(func):
            def wrp2(*args, **kargs):
                t0 = time()
                res = func(*args, **kargs)
                dt = time() - t0
                self._report(name, dt)
                return res
            return wrp2
        return wrapping


class Measure:
    """ ContextWrapper, should not be instantiated directly """
    def __init__(self, timer: Timer, name: str):
       self.timer = timer
       self.name = name
       self.t0 = None

    def __enter__(self):
        self.t0 = time()

    def __exit__(self, a, b, c):
        dt = time() - self.t0
        self.timer._report(self.name, dt)


_global_timer = Timer('global')
measure = _global_timer.measure
wrap = _global_timer.wrap
print_report = _global_timer.print_report
