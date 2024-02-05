"""
Class for reading experimental logs. 
"""
import os
import pickle
from functools import cmp_to_key

import torch

from .dir_utils import get_latest_run_id
from ..utils.plot_util import *


def job_dir_from_run(run_dir):
    # assume that run_dir does not have any trailing /
    dirname = os.path.dirname(run_dir)
    return os.path.basename(dirname)


def load_args(run_dir):
    args_loc = os.path.join(run_dir, 'args.pkl')
    with open(args_loc, 'rb') as args_file:
        return pickle.load(args_file)


def perform_reduce_op(pd_series, reduce_op, weight=None):
    if reduce_op == 'mean':
        return pd_series.mean()
    if reduce_op == 'max':
        return pd_series.max()
    if reduce_op == 'min':
        return pd_series.min()
    if reduce_op == 'weighted_mean':
        return (pd_series * weight).sum() / weight.sum()


def resume_killed(outer_dir):
    run_id = get_latest_run_id(outer_dir) - 1
    while True:
        run_reader = RunReader(os.path.join(outer_dir, 'run_%d' % (run_id)))
        ckpt = run_reader.load_checkpoint(None, latest=True)
        if ckpt is not None:
            return run_reader
        run_id -= 1
        if run_id < 0:
            return None


class RunReader(object):
    def __init__(
            self,
            run_dir):
        """ 
        if outer_dir isn't none, we override run_dir with latest from outer_dir
        """
        self.run_dir = run_dir
        self.checkpoint_dir = os.path.join(self.run_dir, 'checkpoints')
        self.data_dict_dir = os.path.join(self.run_dir, 'data_dicts')
        self.saved_tensor_dir = os.path.join(self.run_dir, 'saved_tensors')

    def read_args(self):
        self.args = load_args(self.run_dir)
        return self.args

    def load_checkpoint(
            self,
            ckpt_path,
            latest=False):
        if latest:
            for fname in os.listdir(self.checkpoint_dir):
                if 'latest_' in fname:
                    latest_fname = os.path.join(self.checkpoint_dir, fname)
                    ckpt_suff = fname.split('latest_')[-1].split('.pth')[0]
                    return torch.load(latest_fname), ckpt_suff
        else:
            full_path = os.path.join(self.checkpoint_dir, ckpt_path)
            ckpt_suff = ckpt_path.split('.pth')[0]
            return torch.load(full_path), ckpt_suff

    def load_data_dict(
            self,
            dict_id='default'):
        dict_path = os.path.join(self.data_dict_dir, '{}.pkl'.format(dict_id))
        return pd.read_pickle(dict_path)

    def load_saved_tensor(self, tensor_path):
        if not tensor_path.endswith('.pth'):
            tensor_path += '.pth'
        return torch.load(os.path.join(self.saved_tensor_dir, tensor_path), map_location='cpu')

    def load_data_dicts(self):
        data_dicts = {}
        for fname in os.listdir(self.data_dict_dir):
            dict_path = os.path.join(self.data_dict_dir, fname)
            dict_id = fname.split('.pkl')[0]
            data_dicts[dict_id] = pd.read_pickle(dict_path)
        return data_dicts

    def obtain_summary_stats(self, stats, dict_id='default'):
        try:
            data_dict = self.load_data_dict(dict_id=dict_id)
        except FileNotFoundError:
            print('Data dict not found for %s' % (self.run_dir))
            return None
        reduce_data_dict = {}
        for item in stats:
            if len(item) == 2:
                col_name, reduce_op = item
                reduce_data_dict[col_name] = perform_reduce_op(data_dict[col_name], reduce_op)
            elif len(item) == 3:
                col_name, reduce_op, weight_col = item
                reduce_data_dict[col_name] = perform_reduce_op(data_dict[col_name], reduce_op,
                                                               weight=data_dict[weight_col])
        return reduce_data_dict


class ExperimentLogReader(object):
    def __init__(
            self,
            outer_dir,
            run_dirs=[]):
        """run_dirs should be a list of absolute paths (not relative to outer_dir)"""

        self.outer_dir = outer_dir
        self.runs_init = False
        self.run_dirs = None
        if len(run_dirs) > 0:
            self.run_dirs = run_dirs
            self.runs_init = True
            self.run_readers = [RunReader(run_dir) for run_dir in self.run_dirs]
            self.set_labels()
            self.cmap = get_cmap(len(self.run_dirs) + 1)

    # TODO: possibly improve on this in the future using some different way
    # to query fix_args?
    def query(
            self,
            job_ids=[],
            fix_args={}
    ):
        """
        job_ids : list of slurm job ids
        fix_args: args to fix, if a list is passed in for the arg, then 
            we allow any value in the list
        """

        run_dirs = self.run_dirs
        if not self.runs_init:
            # get all directories corresponding to submitted slurm jobs
            job_dirs = [name for name in os.listdir(self.outer_dir) \
                        if name.isdigit() and os.path.isdir(os.path.join(self.outer_dir, name))]

            if len(job_ids) > 0:
                # filter only the passed in job ids
                job_dirs = [name for name in job_dirs if int(name) in job_ids]

            run_dirs = []
            for job_dir in job_dirs:
                full_dir = os.path.join(self.outer_dir, job_dir)
                curr_dirs = [os.path.join(full_dir, name) for name in os.listdir(full_dir) if
                             os.path.isdir(os.path.join(full_dir, name))]
                run_dirs += curr_dirs
        else:
            # we want to filter by the job ids
            if len(job_ids) > 0:
                run_dirs = [run_dir for run_dir in self.run_dirs if job_dir_from_run(run_dir) in job_ids]

        all_data = [(vars(load_args(run_dir)), run_dir) for run_dir in run_dirs]

        def filter_args(list_elem):
            curr_args = list_elem[0]
            filter_val = True
            for arg in fix_args:
                if arg not in curr_args:
                    filter_val = False
                    break

                if not isinstance(fix_args[arg], list):
                    obj_type = type(fix_args[arg])
                    if obj_type(curr_args[arg]) != fix_args[arg]:
                        filter_val = False
                        break
                else:
                    obj_type = type(fix_args[arg][0])
                    if obj_type(curr_args[arg]) not in fix_args[arg]:
                        filter_val = False
                        break

            return filter_val

        all_data = list(filter(filter_args, all_data))
        new_runs = [list_elem[1] for list_elem in all_data]

        return ExperimentLogReader(self.outer_dir, new_runs)

    def sort_runs(
            self,
            sort_by):
        """
        sort_by : args to sort, if a tuple (arg_name, type) is 
            passed in, then cast to that type first before sorting
        """
        run_dirs = self.run_dirs
        all_data = [(vars(load_args(run_dir)), run_dir) for run_dir in run_dirs]

        # now we want to sort using args in sort_by

        def cmp_func(data1, data2):
            args1 = data1[0]
            args2 = data2[0]
            for sort_key in sort_by:
                sort_type = str
                if isinstance(sort_key, tuple):
                    sort_type = sort_key[1]
                    sort_key = sort_key[0]
                s1 = sort_key in args1
                s2 = sort_key in args2
                if s1 and not s2:
                    return 1
                if s2 and not s1:
                    return -1
                if s1 and s2:
                    if sort_type(args1[sort_key]) > sort_type(args2[sort_key]):
                        return 1
                    if sort_type(args1[sort_key]) < sort_type(args2[sort_key]):
                        return -1

            return 0

        sorted_data = sorted(all_data, key=cmp_to_key(cmp_func))
        new_runs = [list_elem[1] for list_elem in sorted_data]

        self.run_dirs = new_runs
        self.run_readers = [RunReader(run_dir) for run_dir in new_runs]
        self.set_labels()
        self.cmap = get_cmap(len(self.run_dirs) + 1)

    def combine_runs(
            self,
            run_dirs):
        self.run_dirs += run_dirs
        self.run_readers += [RunReader(run_dir) for run_dir in run_dirs]
        self.set_labels()
        self.cmap = get_cmap(len(self.run_dirs) + 1)

    # access the data dict for low level things
    def get_data_dicts(
            self,
            dict_id='default'):
        dict_list = []
        for run_reader in self.run_readers:
            if dict_id is not None:
                try:
                    dict_list.append(run_reader.load_data_dict(dict_id))
                except FileNotFoundError:
                    print('Data dict not found for %s' % (run_reader.run_dir))
            else:
                dict_list.append(run_reader.load_data_dicts())
        return dict_list

    # set the labels for all of the runs
    def get_labels(
            self,
            label_args=[],
            manual_labels={},
            display_jobs=False):
        """ 
        set default labels which we use throughout plotting, etc.
        label_args: list of arguments to set labels with
        manual_labels: override labels at certain indices
        """
        labels_to_use = []
        if len(label_args) == 0:
            run_dirs = [os.path.join(*run_dir.split('/')[-2:]) for run_dir in self.run_dirs]
            labels_to_use = [run_dir for run_dir in self.run_dirs]
        else:
            all_args = [vars(load_args(run_dir)) for run_dir in self.run_dirs]
            if display_jobs:
                curr_labels = ['{}_'.format(job_dir_from_run(run_dir)) \
                               for run_dir in self.run_dirs]
            else:
                curr_labels = ['' for run_dir in self.run_dirs]
            for ind, argset in enumerate(all_args):
                for arg in label_args:
                    if isinstance(arg, tuple):
                        arg_name = arg[1]
                        arg = arg[0]
                    else:
                        arg_name = arg

                    if arg in argset:
                        curr_labels[ind] += '{}={},'.format(arg_name, argset[arg])

            labels_to_use = curr_labels
        for ind in manual_labels:
            labels_to_use[ind] = manual_labels[ind]

        return labels_to_use

    def set_labels(
            self,
            **kwargs):
        self.labels = self.get_labels(**kwargs)
        return self.labels

    def print_summary_stats(
            self,
            stats,
            dict_id='default',
            label_args=[],
            manual_labels={}
    ):
        """
        stats will be a combination of summary stats to print
        in the form (stat name, reduce operation)
        """
        summaries = []
        if len(label_args) > 0 or len(manual_labels) > 0:
            labels_to_use = self.get_labels(label_args=label_args, manual_labels=manual_labels)
        else:
            labels_to_use = self.labels

        filtered_labels = []
        for ind, run_reader in enumerate(self.run_readers):
            summary = run_reader.obtain_summary_stats(stats, dict_id=dict_id)
            if summary is not None:
                summaries.append(summary)
                filtered_labels.append(labels_to_use[ind])

        for label, summary in zip(filtered_labels, summaries):
            summary_str = '{} : '.format(label)
            for item in stats:
                stat_name = item[0]
                summary_str += '{}: {} '.format(stat_name, summary[stat_name])
            print(summary_str)

    # generic plotting function
    def make_plots(
            self,
            x,
            y,
            hparams={},
            dict_id='default',
            label_args=[],
            manual_labels={},
            where=[]):

        if len(label_args) > 0 or len(manual_labels) > 0:
            labels_to_use = self.get_labels(label_args=label_args, manual_labels=manual_labels)
        else:
            labels_to_use = self.labels

        fig, ax = plt.subplots()
        if not isinstance(dict_id, list):
            dict_id = [dict_id]
        cmap = get_cmap((len(self.run_dirs) + 1) * len(dict_id))
        index = 0
        for d_id in dict_id:
            data_dicts = self.get_data_dicts(dict_id=d_id)
            label_to_add = '-' + d_id if len(dict_id) > 1 else ''

            for ind, data_dict in enumerate(data_dicts):
                if any(condition not in labels_to_use[ind] for condition in where):
                    continue
                x_vals = data_dict[x]
                y_vals = data_dict[y]
                if 'alpha' in hparams:
                    y_vals = smooth_vals(y_vals, hparams['alpha'])
                ax.plot(
                    x_vals,
                    y_vals,
                    label=labels_to_use[ind] + label_to_add,
                    marker=None,
                    linestyle='-',
                    color=cmap(index))
                index += 1

            if 'y_scale' in hparams:
                ax.set_yscale(hparams['y_scale'])
            if 'x_scale' in hparams:
                ax.set_xscale(hparams['x_scale'])

            if 'min_x' in hparams and 'max_x' in hparams:
                ax.set_xlim(hparams['min_x'], hparams['max_x'])
            if 'min_y' in hparams and 'max_y' in hparams:
                ax.set_ylim(hparams['min_y'], hparams['max_y'])

            if hparams.get('show_legend', True):
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)
            ax.set_title('{} vs {}'.format(x, y))
            ax.set_xlabel(x)
            ax.set_ylabel(y)
