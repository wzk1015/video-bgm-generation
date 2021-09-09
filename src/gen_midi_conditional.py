import sys
import os

import math
import time
import glob
import json
import numpy as np

import torch
import argparse

import sys

sys.path.append("../lpd_dataset/")

from numpy2midi_mix import numpy2midi
from dictionary_mix import genre

num_songs = 3

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
print("device", os.environ['CUDA_VISIBLE_DEVICES'])


def generate():
    # path
    from model_encoder import ModelForInference

    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-c', '--ckpt', default="../exp/loss_8_params.pt")
    parser.add_argument('-f', '--files', default="../inference/vlog_short.npz")
    args = parser.parse_args()

    path_saved_ckpt = args.ckpt
    filelist = glob.glob(args.files)
    # outdir

    decoder_n_class = [18, 3, 18, 129, 18, 6, 20, 102, 4865]
    init_n_token = [7, 1, 6]

    # log

    # init model
    net = torch.nn.DataParallel(ModelForInference(decoder_n_class, init_n_token))

    # load model
    print('[*] load model from:', path_saved_ckpt)
    if torch.cuda.is_available():
        net.cuda()
        net.eval()
        net.load_state_dict(torch.load(path_saved_ckpt))
    else:
        net.eval()
        net.load_state_dict(torch.load(path_saved_ckpt, map_location=torch.device('cpu')))

    if len(filelist) == 0:
        raise RuntimeError('no npz file in ' + filelist)

    ###
    genre = {
        'Pop'       : 5,
        'Rock'      : 6,
    }

    for file_name in filelist:
        # gen
        start_time = time.time()
        song_time_list = []
        words_len_list = []

        cnt_tokens_all = 0

        for g, value in genre.items():
            sidx = 0

            while sidx < num_songs:
                try:
                    print("new song")
                    start_time = time.time()
                    vlog_npz = np.load(file_name)['input']
                    pre_init = np.array([
                        [value, 0, 0],  # genre
                        [0, 0, 0],  # key
                        [0, 0, 1],
                        [0, 0, 2],
                        [0, 0, 3],
                        [0, 0, 4],
                        [0, 0, 5],
                    ])

                    C = 0.7
                    vlog_npz = vlog_npz[vlog_npz[:, 2] != 1]
                    print(vlog_npz)

                    res, err_note_number_list, err_beat_number_list = net(vlog=vlog_npz,
                                                                          o_den_track_list=[1, 2, 3], pre_init=pre_init,
                                                                          C=C)
                    # res = net(None, condition=False)
                    print("err_note_number_list", err_note_number_list)
                    print("err_beat_number_list", err_beat_number_list)
                    print("err_note_number_list", np.mean(err_note_number_list))
                    print("err_beat_number_list", np.mean(err_beat_number_list))

                    # res = net(None, condition=True, vlog=vlog_npz, o_den_track_list=[1, 2, 3])
                    # result = res[:, [1, 0, 2, 3, 4, 5, 6]].astype(np.int32)

                    numpy2midi(f"{file_name}_{g}_{sidx}", res[:, [1, 0, 2, 3, 4, 5, 6]].astype(np.int32))
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

    # print('ave token time:', sum(words_len_list) / sum(song_time_list))
    # print('ave song time:', np.mean(song_time_list))
    #
    # runtime_result = {
    #     'song_time': song_time_list,
    #     'words_len_list': words_len_list,
    #     'ave token time:': sum(words_len_list) / sum(song_time_list),
    #     'ave song time': float(np.mean(song_time_list)),
    # }
    #
    # with open('runtime_stats.json', 'w') as f:
    #     json.dump(runtime_result, f)


if __name__ == '__main__':
    print("inference")
    generate()
