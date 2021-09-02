import matplotlib.pyplot as plt
from sklearn import manifold, datasets
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
sys.path.append("/home/brd/music_generation/compound-word-transformer/lpd_dataset/mix/")

from numpy2midi_mix import numpy2midi

init_dictionary ={
"instr_type": {
    'None': 0,
    'Drums': 1,
    'Piano': 2,
    'Guitar': 3,
    'Bass': 4,
    'Strings': 5,
},
"key": {
    "None": 0,
    'C': 1,
    'C#': 2,
    'D': 3,
    'D#': 4,
    'E': 5,
    'F': 6,
    'F#': 7,
    'G': 8,
    'G#': 9,
    'A': 10,
    'A#': 11,
    'B': 12,
    'c': 13,
    'c#': 14,
    'd': 15,
    'd#': 16,
    'e': 17,
    'f': 18,
    'f#': 19,
    'g': 20,
    'g#': 21,
    'a': 22,
    'a#': 23,
    'b': 24,
},
"genre":{
    "None":0,
    'classic': 1,
    'country': 2,
    'dance': 3,
    'electronic': 4,
    'pop': 5,
    'rock': 6,
}
}

genre = {
    'classic': 1,
    # 'country': 2,
    # 'dance': 3,
    # 'electronic': 4,
    # 'pop': 5,
    # 'rock': 6,
}
used_instr = [[1,2,3,4,5]]

# MODE = 'train'
MODE = 'inference'

path_gendir = 'original_cp_gen_midis'
if not os.path.exists(path_gendir):
    os.mkdir(path_gendir)
num_songs = 3

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
print("device", os.environ['CUDA_VISIBLE_DEVICES'])

def generate():
    # path
    from model_encoder import ModelForInference
    path_saved_ckpt = "/home/brd/music_generation/compound-word-transformer/workspace/uncond/jzr/exp/encoder_mix_large_dataset_b_den_init_token/loss_11_params.pt"

    # outdir
    os.makedirs(path_gendir, exist_ok=True)

    # config
    # n_class = []
    # for key in event2word.keys():
    #     n_class.append(len(dictionary[0][key]))
    #
    # n_class = n_class[2:]
    # encoder_n_class = [17, 4, 12, 35]
    decoder_n_class = [18,3,18,129,18,6,20]
    init_n_token = [7,25,6]

    # log
    # log('num of classes:', n_class)

    # init model
    net = torch.nn.DataParallel(ModelForInference(decoder_n_class, init_n_token, is_training=False))
    net.cuda()
    net.eval()

    # load model
    print('[*] load model from:', path_saved_ckpt)
    net.load_state_dict(torch.load(path_saved_ckpt))

    # gen
    start_time = time.time()
    song_time_list = []
    words_len_list = []

    cnt_tokens_all = 0
    for g, value in genre.items():
        sidx = 0
        while sidx < num_songs:
            try:
                start_time = time.time()
                print('current idx:', sidx)
                path_outfile = os.path.join(path_gendir, 'get_{}.mid'.format(str(sidx)))
                vlog_npz=np.load("./wzk_v2.npz")['input']
                # import ipdb; ipdb.set_trace()

                # pre_init = np.array([
                #     [value, 0, 0],  # genre
                #     [0, 1, 0],  # key
                #     [0, 0, 1],
                #     [0, 0, 2],
                #     [0, 0, 3],
                #     [0, 0, 4],
                #     [0, 0, 5],
                # ])

                pre_init = np.array([
                    [0, 0, 0],  # genre
                    [1, 1, 1],  # key
                    [2, 2, 2],
                    [3, 3, 3],
                    [4, 4, 4],
                    [5, 5, 5],
                    [6, 6, 5],
                    [6, 7, 5],
                    [6, 8, 5],
                    [6, 9, 5],
                    [6, 10, 5],
                    [6, 11, 5],
                    [6, 12, 5],
                    [6, 13, 5],
                    [6, 14, 5],
                    [6, 15, 5],
                    [6, 16, 5],
                    [6, 17, 5],
                    [6, 18, 5],
                    [6, 19, 5],
                    [6, 20, 5],
                    [6, 21, 5],
                    [6, 22, 5],
                    [6, 23, 5],
                    [6, 24, 5],


                ])

                emb = net(None, condition=True, vlog=vlog_npz, o_den_track_list=[1, 2, 3], pre_init=pre_init, visualize=True)

                emb_list = []
                for e in emb:
                    emb_list.append(e.cpu().detach().numpy())
                n_neighbors = 10
                n_components = 2

                # # 绘制S型曲线的3D图像
                # ax = fig.add_subplot(211, projection='3d')  # 创建子图
                # ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=color, cmap=plt.cm.Spectral)  # 绘制散点图，为不同标签的点赋予不同的颜色
                # ax.set_title('Original S-Curve', fontsize=14)
                # ax.view_init(4, -72)  # 初始化视角
                # emb_note=emb[emb[:,1]==2]
                # print(emb.shape)
                # t-SNE的降维与可视化

                # for k, v in DECODER_DIMENSION.items():
                ts = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
                # 训练模型
                y = ts.fit_transform(emb_list[2])
                # 创建自定义图像
                fig = plt.figure(figsize=(8, 8))  # 指定图像的宽和高
                plt.suptitle("Dimensionality Reduction and Visualization of S-Curve Data ", fontsize=14)  # 自定义图像名称
                ax1 = fig.add_subplot(1, 1, 1)
                plt.scatter(y[:, 0], y[:, 1], c=pre_init[:, 2], cmap=plt.cm.Spectral)
                # for i in range(len(x)):
                for i, k in enumerate(init_dictionary['instr_type'].keys()):
                    plt.annotate(k, xy=(y[i, 0], y[i, 1]), xytext=(y[i, 0] + 2.0, y[i, 1] + 2.0))
                ax1.set_title('t-SNE Curve instr_type', fontsize=14)
                # ax1.view_init(4, -72)
                plt.colorbar()
                # 显示图像
                plt.savefig('./visualize_single_dimension_instr_type.png')
                exit()
                # res = net(None, condition=False)
                # import ipdb; ipdb.set_trace()
                # zero = np.zeros((res.shape[0], 1))
                # result = np.concatenate([res, zero], axis=1)

                # numpy2midi("init12345_" + g + "_wzk_vlog_beat_enhance1_track123_"+str(sidx), res[:, [1, 0, 2, 3, 4, 5, 6]].astype(np.int32))

                # res = net(dictionary)
                # write_midi(res, path_outfile, word2event)

                # song_time = time.time() - start_time
                # word_len = len(res)
                # print('song time:', song_time)
                # print('word_len:', word_len)
                # words_len_list.append(word_len)
                # song_time_list.append(song_time)

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
    # -- training -- #
    # vlog_npz=np.load("./vlog_360p.npz")
    # import ipdb; ipdb.set_trace()
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
