"""
Class for writing experimental logs. 
"""
import os
import pickle

import pandas as pd
import torch

from ..utils.metric import create_metric


class ExperimentLogWriter(object):
    def __init__(self, dir):
        slurm_dir = dir

        if not os.path.exists(slurm_dir):
            os.makedirs(slurm_dir)

        self.save_dir = slurm_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.log_loc = os.path.join(self.save_dir, 'log.txt')

        self.data_dict_dir = os.path.join(self.save_dir, 'data_dicts')
        if not os.path.exists(self.data_dict_dir):
            os.makedirs(self.data_dict_dir)

        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.saved_tensor_dir = os.path.join(self.save_dir, 'saved_tensors')
        if not os.path.exists(self.saved_tensor_dir):
            os.makedirs(self.saved_tensor_dir)

        self.data_dicts = {}  # data for storing
        self.metrics = {}

    def save_args(
            self,
            args):
        self.args_loc = os.path.join(self.save_dir, 'args.pkl')
        pickle.dump(args, open(self.args_loc, 'wb'))

        # also write the args to the log file
        args_dict = vars(args)
        with open(self.log_loc, 'a') as log_file:
            for arg in sorted(args_dict.keys()):
                log_file.write(
                    '{:20} : {}\n'.format(arg, args_dict[arg]))

    # logging
    def log(
            self,
            log_str):
        with open(self.log_loc, 'a') as log_file:
            log_file.write('{}\n'.format(log_str))

    # handle metric storing and updating
    def add_metric(
            self,
            metric_name,
            metric_type='avg'):
        self.metrics[metric_name] = create_metric(metric_type)

    def get_metric(
            self,
            metric_name):
        return self.metrics[metric_name].val()

    def update_metric(
            self,
            metric_name,
            update_val,
            **update_kwargs):
        return self.metrics[metric_name].update(update_val, **update_kwargs)

    def reset_metric(
            self,
            metric_name):
        self.metrics[metric_name].reset()

    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset()

    # handle data dictionary creation and saving
    def create_data_dict(
            self,
            col_names,
            dict_id='default'):
        df = pd.DataFrame({col_name: [] for col_name in col_names})
        self.data_dicts[dict_id] = df

    def update_data_dict(
            self,
            update_dict,
            dict_id='default'):
        df = pd.Series(update_dict)
        self.data_dicts[dict_id] = self.data_dicts[dict_id].append(df, ignore_index=True)

    def save_data_dict(
            self,
            dict_id='default'):
        dict_path = os.path.join(self.data_dict_dir, '{}.csv'.format(dict_id))
        self.data_dicts[dict_id].to_csv(dict_path, float_format='%.3f')

    def save_data_dicts(
            self):
        for dict_id in self.data_dicts:
            self.save_data_dict(dict_id)

    # handle model checkpointing
    def ckpt_model(
            self,
            to_ckpt,
            index,
            is_latest=False):
        ckpt_str = 'latest_' if is_latest else ''
        ckpt_loc = os.path.join(self.checkpoint_dir, '{}{}.pth'.format(ckpt_str, index))

        # remove the previous if it is the latest
        if is_latest:
            for fname in os.listdir(self.checkpoint_dir):
                if 'latest_' in fname:
                    os.remove(os.path.join(self.checkpoint_dir, fname))

        torch.save(to_ckpt, ckpt_loc)

    def save_tensor(self, to_save, filename):
        if not filename.endswith('.pth'):
            filename += '.pth'
        tensor_loc = os.path.join(self.saved_tensor_dir, filename)
        torch.save(to_save, tensor_loc)
