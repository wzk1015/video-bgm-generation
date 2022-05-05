import sys
import os

import time
import glob
import numpy as np

import torch
import argparse

sys.path.append("../dataset/")

from numpy2midi_mix import numpy2midi
from model import CMT


def cal_control_error(err_note_number_list, err_beat_number_list):
    print("err_note_number_list", err_note_number_list)
    print("err_beat_number_list", err_beat_number_list)
    print("strength control error", np.mean(err_note_number_list) / 1.83)
    print("density control error", np.mean(err_beat_number_list) / 10.90)


def generate():
    # path
    parser = argparse.ArgumentParser(description="Args for generating background music")
    parser.add_argument('-c', '--ckpt', default="../exp/loss_8_params.pt", help="Model checkpoint to be loaded")
    parser.add_argument('-f', '--files', required=True, help="Input npz file of a video")
    parser.add_argument('-g', '--gpus', help="Id of gpu. Only ONE gpu is needed")
    parser.add_argument('-n', '--num_songs', default=1, help="Number of generated songs")
    args = parser.parse_args()
    
    num_songs = int(args.num_songs)

    if args.gpus is not None:
        if not args.gpus.isnumeric():
            raise RuntimeError('Only 1 GPU is needed for inference')
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    path_saved_ckpt = args.ckpt
    filelist = glob.glob(args.files)


    # change this if using another training set (see the output of decoder_n_class in train.py)
    decoder_n_class = [18, 3, 18, 129, 18, 6, 20, 102, 5025] 
    init_n_token = [7, 1, 6]


    # init model
    net = torch.nn.DataParallel(CMT(decoder_n_class, init_n_token))

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
        raise RuntimeError('no npz file in ' + str(filelist))

    for file_name in filelist:
        # gen
        start_time = time.time()
        song_time_list = []
        words_len_list = []

        sidx = 0

        while sidx < num_songs:
            try:
                print("new song")
                start_time = time.time()
                vlog_npz = np.load(file_name)['input']

                vlog_npz = vlog_npz[vlog_npz[:, 2] != 1]
                print(vlog_npz)

                res, err_note_number_list, err_beat_number_list = net(is_train=False, vlog=vlog_npz, C=0.7)

                cal_control_error(err_note_number_list, err_beat_number_list)

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


if __name__ == '__main__':
    print("inference")
    generate()
