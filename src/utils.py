import math
import numpy as np
import torch
import torch.nn as nn

import os
import time
import collections
import matplotlib.pyplot as plt
import logging

flog = None

def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

class Saver(object):
    def __init__(
            self,
            exp_dir,
            mode='w',
            debug=False):

        self.exp_dir = exp_dir
        self.init_time = time.time()
        self.global_step = 0
        self.debug = debug

        # makedirs
        os.makedirs(exp_dir, exist_ok=True)

        # logging config
        path_logger = os.path.join(exp_dir, 'log.txt')
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(message)s',
            filename=path_logger,
            filemode=mode)
        self.logger = logging.getLogger('training monitor')

    def add_summary_msg(self, msg):
        if self.debug:
            return
        self.logger.debug(msg)

    def add_summary(
            self,
            key,
            val,
            step=None,
            cur_time=None):
        if self.debug:
            return
        if cur_time is None:
            cur_time = time.time() - self.init_time
        if step is None:
            step = self.global_step

        # write msg (key, val, step, time)
        if isinstance(val, float):
            msg_str = '{:10s} | {:.10f} | {:10d} | {}'.format(
                key,
                val,
                step,
                cur_time
            )
        else:
            msg_str = '{:10s} | {} | {:10d} | {}'.format(
                key,
                val,
                step,
                cur_time
            )

        self.logger.debug(msg_str)

    def save_model(
            self,
            model,
            optimizer=None,
            outdir=None,
            name='model'):
        if self.debug:
            return
        if outdir is None:
            outdir = self.exp_dir
        print(' [*] saving model to {}, name: {}'.format(outdir, name))
        # torch.save(model, os.path.join(outdir, name+'.pt'))
        torch.save(model.state_dict(), os.path.join(outdir, name + '_params.pt'))

        if optimizer is not None:
            torch.save(optimizer.state_dict(), os.path.join(outdir, name + '_opt.pt'))

    def load_model(
            self,
            path_exp,
            device='cpu',
            name='model.pt'):

        path_pt = os.path.join(path_exp, name)
        print(' [*] restoring model from', path_pt)
        model = torch.load(path_pt, map_location=torch.device(device))
        return model

    def global_step_increment(self):
        self.global_step += 1


def make_loss_report(
        path_log,
        path_figure='loss.png',
        dpi=100):
    # load logfile
    monitor_vals = collections.defaultdict(list)
    with open(path_log, 'r') as f:
        for line in f:
            try:
                line = line.strip()
                key, val, step, acc_time = line.split(' | ')
                monitor_vals[key].append((float(val), int(step), acc_time))
            except:
                continue

    # collect
    step_train = [item[1] for item in monitor_vals['train loss']]
    vals_train = [item[0] for item in monitor_vals['train loss']]

    step_valid = [item[1] for item in monitor_vals['valid loss']]
    vals_valid = [item[0] for item in monitor_vals['valid loss']]

    x_min = step_valid[np.argmin(vals_valid)]
    y_min = min(vals_valid)

    # plot
    fig = plt.figure(dpi=dpi)
    plt.title('training process')
    plt.plot(step_train, vals_train, label='train')
    plt.plot(step_valid, vals_valid, label='valid')
    plt.yscale('log')
    plt.plot([x_min], [y_min], 'ro')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(path_figure)


def log(*args, **kwargs):
    print(*args, **kwargs)
    if flog is not None:
        print(*args, file=flog, flush=True)