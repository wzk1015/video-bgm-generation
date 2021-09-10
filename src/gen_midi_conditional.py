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


def cal_control_error(err_note_number_list, err_beat_number_list):
    # number of notes per simu-note
    # o_dens: 1, 7868576 / 14468678, 54.38%
    # o_dens: 2, 3334852 / 14468678, 23.05%
    # o_dens: 3, 1934401 / 14468678, 13.37%
    # o_dens: 4, 835012 / 14468678, 5.77%
    # o_dens: 5, 308613 / 14468678, 2.13%
    # o_dens: 6, 130653 / 14468678, 0.90%
    # o_dens: 7, 36176 / 14468678, 0.25%
    # o_dens: 8, 13118 / 14468678, 0.09%
    # o_dens: 9, 4381 / 14468678, 0.03%
    # o_dens: 10, 1694 / 14468678, 0.01%
    # o_dens: 11, 549 / 14468678, 0.00%
    # o_dens: 12, 383 / 14468678, 0.00%
    # o_dens: 13, 92 / 14468678, 0.00%
    # o_dens: 14, 33 / 14468678, 0.00%
    # o_dens: 15, 59 / 14468678, 0.00%
    # o_dens: 16, 21 / 14468678, 0.00%
    # o_dens: 17, 1 / 14468678, 0.00%
    # o_dens: 18, 61 / 14468678, 0.00%
    # o_dens: 19, 3 / 14468678, 0.00%
    # average: 1.83

    # number of simu-notes per bar
    # b_dens: 1, 12998 / 665612, 1.95%
    # b_dens: 2, 6042 / 665612, 0.91%
    # b_dens: 3, 4340 / 665612, 0.65%
    # b_dens: 4, 4978 / 665612, 0.75%
    # b_dens: 5, 12847 / 665612, 1.93%
    # b_dens: 6, 15565 / 665612, 2.34%
    # b_dens: 7, 22219 / 665612, 3.34%
    # b_dens: 8, 29463 / 665612, 4.43%
    # b_dens: 9, 180284 / 665612, 27.09%
    # b_dens: 10, 78078 / 665612, 11.73%
    # b_dens: 11, 52481 / 665612, 7.88%
    # b_dens: 12, 34962 / 665612, 5.25%
    # b_dens: 13, 46884 / 665612, 7.04%
    # b_dens: 14, 29673 / 665612, 4.46%
    # b_dens: 15, 23716 / 665612, 3.56%
    # b_dens: 16, 18494 / 665612, 2.78%
    # b_dens: 17, 92588 / 665612, 13.91%
    # average: 10.90

    print("err_note_number_list", err_note_number_list)
    print("err_beat_number_list", err_beat_number_list)
    print("strength control error", np.mean(err_note_number_list) / 1.83)
    print("density control error", np.mean(err_beat_number_list) / 10.90)


def generate():
    # path
    from model_encoder import ModelForInference

    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('-c', '--ckpt', default="../exp/loss_8_params.pt")
    parser.add_argument('-f', '--files', default="../inference/wzk.npz")
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
                    cal_control_error(err_note_number_list, err_beat_number_list)


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
