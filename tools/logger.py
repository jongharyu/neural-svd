import os
from collections import OrderedDict

from tensorboardX import SummaryWriter
from torch import Tensor

from .plotter import Plotter


class Logger(object):
    def __init__(self, log_dir, tensorboard=True, matplotlib=True):

        self.reset(log_dir, tensorboard, matplotlib)

    def reset(self, log_dir=None, tensorboard=True, matplotlib=True):

        if log_dir is not None: self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir=self.log_dir) if tensorboard else None
        self.plotter = Plotter() if matplotlib else None
        self.counter = OrderedDict()

    def update_scalers(self, ordered_dict):

        for key, value in ordered_dict.items():
            if isinstance(value, Tensor):
                # ipdb.set_trace()
                try:
                    ordered_dict[key] = value.item()
                except:
                    ordered_dict[key] = value.mean().item()
                # ipdb.set_trace()
            if self.counter.get(key) is None:
                self.counter[key] = 1
            else:
                self.counter[key] += 1

            if self.writer:
                try:
                    self.writer.add_scalar(key, value, self.counter[key])
                except:
                    self.writer.add_scalar(key, value.mean(), self.counter[key])

        if self.plotter:
            self.plotter.update(ordered_dict)
            self.plotter.save(os.path.join(self.log_dir, 'plotter.svg'))
