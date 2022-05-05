#/usr/bin/env python
# -*- coding: UTF-8 -*-
import json
import os
import math
import argparse

import numpy as np

from dictionary_mix import preset_event2word
from stat_mix import vbeat_weight_percentile, fmpb_percentile

RESOLUTION = 16
DIMENSION = {
    'beat'    : 0,
    'density' : 1,
    'strength': 2,
    'i_beat'  : 3,
    'n_beat'  : 4,
    'p_beat'  : 5,
}
N_DIMENSION = len(DIMENSION)


def _cal_density(flow_magnitude):
    for i, percentile in enumerate(fmpb_percentile):
        if flow_magnitude < percentile:
            return i
    return len(fmpb_percentile)


def _cal_strength(weight):
    for i, percentile in enumerate(vbeat_weight_percentile):
        if weight < percentile:
            return i
    return len(vbeat_weight_percentile)


def _get_beat_token(beat, strength, i_beat, n_beat):
    l = [[0] * N_DIMENSION]
    l[0][DIMENSION['beat']] = preset_event2word['beat']['Beat_%d' % beat]
    l[0][DIMENSION['strength']] = strength
    l[0][DIMENSION['i_beat']] = i_beat
    l[0][DIMENSION['n_beat']] = n_beat
    l[0][DIMENSION['p_beat']] = round(float(i_beat) / n_beat * 100) + 1
    return l


def _get_bar_token(density, i_beat, n_beat):
    l = [[0] * N_DIMENSION]
    l[0][DIMENSION['beat']] = preset_event2word['beat']['Bar']
    l[0][DIMENSION['density']] = density + 1
    l[0][DIMENSION['i_beat']] = i_beat
    l[0][DIMENSION['n_beat']] = n_beat
    l[0][DIMENSION['p_beat']] = round(float(i_beat) / n_beat * 100) + 1
    return l


def metadata2numpy(metadata):
    vbeats = metadata['vbeats']
    fmpb = metadata['flow_magnitude_per_bar']
    n_beat = int(math.ceil(float(metadata['duration']) / 60 * float(metadata['tempo']) * 4))

    n_bars = 0  # 已添加 bar token 个数
    l = []
    for vbeat in vbeats:
        # add bar token
        while int(vbeat['bar']) >= n_bars:
            i_beat = n_bars * RESOLUTION
            l += _get_bar_token(density=_cal_density(fmpb[n_bars]), i_beat=i_beat, n_beat=n_beat)
            n_bars += 1
        # add beat token
        i_beat = int(vbeat['bar']) * RESOLUTION + int(vbeat['tick'])
        l += _get_beat_token(beat=int(vbeat['tick']), strength=_cal_strength(vbeat['weight']), i_beat=i_beat,
                             n_beat=n_beat)
    # add empty bars
    while n_bars < len(fmpb):
        i_beat = n_bars * RESOLUTION
        l += _get_bar_token(density=_cal_density(fmpb[n_bars]), i_beat=i_beat, n_beat=n_beat)
        n_bars += 1

    return np.asarray(l, dtype=int)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default="../../inference/")
    parser.add_argument('--video', default="final_640.mp4")
    parser.add_argument('--metadata', default="metadata.json")
    args = parser.parse_args()

    video_name = os.path.basename(args.video)

    with open(args.metadata) as f:
        metadata = json.load(f)

    target_path = os.path.join(args.out_dir, video_name.replace('.mp4', '.npz'))

    print('processing to save to %s' % target_path)
    input_numpy = metadata2numpy(metadata)
    np.savez(target_path, input=input_numpy)
    print("saved to " + str(target_path))