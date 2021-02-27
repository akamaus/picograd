import logging
import os
import random
from .system import link
from typing import Optional

import numpy
import torch

logger = logging.getLogger('utils')


class LoadStateError(RuntimeError):
    """ Exception to be invoked on deserialization errors """
    pass

class SaveStateError(RuntimeError):
    """ Various problems during state saving"""
    pass


class Storage:
    LAST_CPT = "last"

    def __init__(self, root_dir='experiments', experiment_name=None):
        assert experiment_name is not None

        self._root_dir = root_dir
        self._experiment_name = experiment_name

    @property
    def experiment_dir(self) -> str:
        d = os.path.join(self._root_dir, self._experiment_name)
        return d

    def checkpoint_path(self, cpt_name: str) -> str:
        return os.path.join(self.experiment_dir, "checkpoints", f'{cpt_name}.ckp')

    def save_state(self, model: torch.nn.Module, train_state:dict, cpt_name:str):
        """ Save experiment checkpoint """
        assert isinstance(cpt_name, str), f"checkpoint name should be str, but is {cpt_name}"
        save_path = self.checkpoint_path(cpt_name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        rng_state = {}
        rng_state['random'] = random.getstate()
        rng_state['numpy'] = numpy.random.get_state()
        rng_state['torch'] = torch.get_rng_state()
        if torch.cuda.is_available():
            rng_state['torch.cuda'] = torch.cuda.get_rng_state()

        model_state = model.state_dict()
        for pn, v in model_state.items():
            if torch.isnan(v).sum() > 0:
                raise SaveStateError('Warning, NaNs in loaded model state detected in parameter', pn)

        state = {'model_state': model_state,
                 'train_state': train_state,
                 'meta_parameters': model.meta_parameters,
                 'rng_state': rng_state
                }

        logger.info(f'saving to {save_path}')
        torch.save(state, save_path)

        last_cpt = self.checkpoint_path(self.LAST_CPT)
        if os.path.exists(last_cpt):
            os.unlink(last_cpt)
        link(os.path.abspath(save_path), last_cpt)

    def load_state(self, checkpoint_path:Optional[str]=None, checkpoint_name:Optional[str]=None, model: Optional[torch.nn.Module]=None):
        """ Loads a state from the checkpoint,  tries to instantinate appropriate model if possible """
        assert not (checkpoint_path is not None and checkpoint_name is not None), \
            "checkpoint_path and checkpoint_name must not be specified together"

        if checkpoint_name is not None:
            checkpoint_path = self.checkpoint_path(checkpoint_name)

        if checkpoint_path is None:  # neither is specified, should load last one
            checkpoint_path = self.checkpoint_path(self.LAST_CPT)

        state = torch.load(checkpoint_path)

        if 'meta_parameters' in state:  # new-style checkpoint
            meta = state['meta_parameters']
            name = meta['name']
            args = meta['args']
            if isinstance(model, type):
                model = model(**args)
            elif model is None:
                import models
                logger.info('Instantinating model "%s" from model directory with args %s' % (name, str(args)))
                model = models.model_directory[name](**args)

        logger.info(f'loading weights from {checkpoint_path}')

        if model is None:
            raise LoadStateError('either use new-style checkpoint or pass model object')

        if 'model_state' in state:
            model_state = state['model_state']
        else:
            model_state = state

        logger.info(f'model.meta_parameters = {model.meta_parameters}')

        for v in model_state.values():
            if torch.isnan(v).sum() > 0:
                print('Warning, NaNs in loaded model state detected')

        model.load_state_dict(model_state)
        rng_state = state.get('rng_state')
        if rng_state:
            st = rng_state.get('random')
            if st is not None:
                random.setstate(st)
            st = rng_state.get('numpy')
            if st is not None:
                numpy.random.set_state(st)
            st = rng_state.get('torch')
            if st is not None:
                torch.set_rng_state(st)
            st = rng_state.get('torch.cuda')
            if st is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(st)

        return model, state.get('train_state')
