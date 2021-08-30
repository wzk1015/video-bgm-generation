import sys
import os

import math
import time
import glob
import datetime
import random
import pickle
import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.builders import RecurrentEncoderBuilder
from fast_transformers.masking import TriangularCausalMask

import miditoolkit
from miditoolkit.midi.containers import Marker, Instrument, TempoChange, Note

import sys
sys.path.append("../lpd_dataset/")

from numpy2midi_mix import numpy2midi


# MODE = 'train'
MODE = 'inference'

num_songs = 3

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
print("device", os.environ['CUDA_VISIBLE_DEVICES'])



def generate():
    # path
    from model_encoder import ModelForInference
    path_saved_ckpt = "../exp/loss_13_params.pt"
    filelist = glob.glob("../inference/*.npz")
    # outdir

    decoder_n_class = [18, 3, 18, 129, 18, 6, 20]

    # log

    # init model
    net = torch.nn.DataParallel(ModelForInference(decoder_n_class, is_training=False))

    # load model
    print('[*] load model from:', path_saved_ckpt)
    if torch.cuda.is_available():
        net.cuda()
        net.eval()
        net.load_state_dict(torch.load(path_saved_ckpt))
    else:
        net.eval()
        net.load_state_dict(torch.load(path_saved_ckpt,map_location=torch.device('cpu')))

    if len(filelist) == 0:
        raise RuntimeError('no npy file in ' + filelist)
    
    for file_name in filelist:
        # gen
        start_time = time.time()
        song_time_list = []
        words_len_list = []

        cnt_tokens_all = 0
        sidx = 0

        while sidx < num_songs:
            try:
                start_time = time.time()
                vlog_npz = np.load(file_name)['input'][:,:3]
                res = net(None, condition=True, vlog=vlog_npz, o_den_track_list=[1, 2, 3])
                result = res[:, [1, 0, 2, 3, 4, 5, 6]].astype(np.int32)
                

                numpy2midi(f"{file_name}_{sidx}", res[:, [1, 0, 2, 3, 4, 5, 6]].astype(np.int32))
                song_time = time.time() - start_time
                word_len = len(res)
                print('song time:', song_time)
                print('word_len:', word_len)
                words_len_list.append(word_len)
                song_time_list.append(song_time)

                sidx += 1
            except KeyboardInterrupt:
                raise ValueError(' [x] terminated.')
        # except:
        #     continue

    print('ave token time:', sum(words_len_list) / sum(song_time_list))
    print('ave song time:', np.mean(song_time_list))

    runtime_result = {
        'song_time': song_time_list,
        'words_len_list': words_len_list,
        'ave token time:': sum(words_len_list) / sum(song_time_list),
        'ave song time': float(np.mean(song_time_list)),
    }

    with open('runtime_stats.json', 'w') as f:
        json.dump(runtime_result, f)


if __name__ == '__main__':
    if MODE == 'train':
        print("training")
        train()

    # -- inference -- #
    elif MODE == 'inference':
        print("inference")
        generate()

    else:
        print("gen with onsets")
        gen_with_onsets(recurrent=False)
