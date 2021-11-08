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


################################################################################
# Sampling
################################################################################
# -- temperature -- #
def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= sum(probs)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


################################################################################
# Model
################################################################################


def network_paras(model):
    # compute only trainable params
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape, self.pe[:, :x.size(1), :].shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BeatPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=20000):
        super(BeatPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, index):
        x = x + self.pe[:, index, :]
        return self.dropout(x)[0]


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