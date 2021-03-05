import logging
import os
import os.path as osp

import random
from .system import link
from typing import Optional, Union, Dict

import numpy
import torch
import torch.nn as nn

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
        d = osp.join(self._root_dir, self._experiment_name)
        return d

    def checkpoint_path(self, cpt_name: str) -> str:
        if osp.sep not in cpt_name:
            path = osp.join(self.experiment_dir, "checkpoints", f'{cpt_name}.ckp')
        else:
            path = cpt_name + '.ckp'
        return path

    def model_name(self, cpt_name: str, model_name: str):
        return cpt_name + '.' + model_name

    def save_model(self, model: nn.Module, save_path: str):
        model_state = model.state_dict()
        for pn, v in model_state.items():
            if torch.isnan(v).sum() > 0:
                raise SaveStateError('Warning, NaNs in loaded model state detected in parameter', pn)

        state = {}
        state['model_state'] = model_state
        if hasattr(model, 'meta_parameters'):
            state['meta_parameters'] = model.meta_parameters

        torch.save(state, save_path)

    def load_model(self, model_path: str, model:Optional[nn.Module]=None):
        state = torch.load(model_path)
        if 'meta_parameters' in state:  # new-style checkpoint
            meta = state['meta_parameters']
            name = meta.get('name')
            args = meta['args']
            if isinstance(model, type):
                model = model(**args)
            elif model is None:
                assert name is not None, 'No model constructor passed, so meta_parameters should have a name'
                import models
                logger.info('Instantinating model "%s" from model directory with args %s' % (name, str(args)))
                model = models.model_directory[name](**args)

        logger.info(f'loading weights from {model_path}')

        if model is None:
            raise LoadStateError('either use new-style checkpoint or pass model object')

        if 'model_state' in state:
            model_state = state['model_state']
        else:
            model_state = state

        if hasattr(model, 'meta_parameters'):
            logger.info(f'model.meta_parameters = {model.meta_parameters}')

        for v in model_state.values():
            if torch.isnan(v).sum() > 0:
                print('Warning, NaNs in loaded model state detected')

        model.load_state_dict(model_state)
        return model

    def save_state(self, models: Union[nn.Module, Dict[str, nn.Module]], train_state:dict, cpt_name:str):
        """ Save experiment checkpoint """
        assert isinstance(cpt_name, str), f"checkpoint name should be str, but is {cpt_name}"
        save_path = self.checkpoint_path(cpt_name)
        os.makedirs(osp.dirname(save_path), exist_ok=True)

        if not isinstance(models, dict):
            models = {'model': models}

        model_fnames = {}
        for mname, model in models.items():
            msave_name = self.model_name(cpt_name, mname)
            model_fnames[mname] = msave_name
            self.save_model(model, self.checkpoint_path(msave_name))

        rng_state = {}
        rng_state['random'] = random.getstate()
        rng_state['numpy'] = numpy.random.get_state()
        rng_state['torch'] = torch.get_rng_state()
        if torch.cuda.is_available():
            rng_state['torch.cuda'] = torch.cuda.get_rng_state()

        state = {
                 'model_fnames': model_fnames,
                 'train_state': train_state,
                 'rng_state': rng_state,
                 'format': 'V2.0'
                }

        logger.info(f'saving to {save_path}')
        torch.save(state, save_path)

        last_cpt = self.checkpoint_path(self.LAST_CPT)
        if osp.exists(last_cpt):
            os.unlink(last_cpt)
        link(osp.abspath(save_path), last_cpt)

    def load_state(self, checkpoint_path:Optional[str]=None, checkpoint_name:Optional[str]=None, model: Optional[nn.Module]=None):
        """ Loads a state from the checkpoint,  tries to instantinate appropriate model if possible """
        assert not (checkpoint_path is not None and checkpoint_name is not None), \
            "checkpoint_path and checkpoint_name must not be specified together"

        if checkpoint_name is not None:
            checkpoint_path = self.checkpoint_path(checkpoint_name)

        if checkpoint_path is None:  # neither is specified, should load last one
            checkpoint_path = self.checkpoint_path(self.LAST_CPT)
            checkpoint_name = self.LAST_CPT
        else:
            checkpoint_name = osp.splitext(osp.basename(checkpoint_path))[0]

        state = torch.load(checkpoint_path)

        fmt = state.get('format', 'V1.0')

        if fmt == 'V1.0':
            model = self.load_model(checkpoint_path, model)  # just read the model data from the same checkpoint
        elif fmt == 'V2.0':
            models = {}
            for m_name, m_fname in state['model_fnames'].items():
                model_path = self.checkpoint_path(osp.join(osp.dirname(checkpoint_path), m_fname))
                m = model.get(m_name) if isinstance(model, dict) else model
                models[m_name] = self.load_model(model_path, model=m)
        else:
            raise LoadStateError('Unknown format', fmt, checkpoint_path)

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

        if len(models) == 1 and 'model' in models:
            models = models['model']

        return models, state.get('train_state')
