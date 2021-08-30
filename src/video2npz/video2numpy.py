import json
import os

import numpy as np
import cv2
import skvideo.io

from optical_flow import dense_optical_flow
from resize_video import resize_video
from midi_utils.midi2numpy_mix import ENCODER_DIMENSION, N_ENCODER_DIMENSION, KEYS, ENCODER_MAX_LEN, RESOLUTION
from midi_utils.dictionary_mix import preset_event2word


method = cv2.calcOpticalFlowFarneback
params = [0.5, 3, 15, 3, 5, 1.2, 0]  # default Farneback's algorithm parameters
to_gray = True
flow_magnitude_percentile = [0.1165, 0.2195, 0.3216, 0.4424, 0.6069, 0.8420, 1.2530, 2.0467, 3.3376, 11.8150]
max_height=360

with open('./vbeats_tempo.json', 'r') as f:
    beats_tempo = json.load(f)


def _cal_density(flow_magnitude):
    for i, percentile in enumerate(flow_magnitude_percentile):
        if flow_magnitude < percentile:
            return i
    return len(flow_magnitude_percentile)


def video2numpy(video_name, in_dir, out_dir='encoder_numpy', key='C', genre='pop'):
    assert key in preset_event2word['key/genre'].keys()
    assert genre in preset_event2word['key/genre'].keys()
    assert video_name in beats_tempo.keys()

    video_path = os.path.join(in_dir, video_name)
    metadata = skvideo.io.ffprobe(video_path)
    height = int(metadata['video']['@height'])
    if height > max_height:
        video_path = resize_video(video_name, in_dir)

    flow_magnitude_per_bar, _ = dense_optical_flow(method, video_path, params, to_gray)

    vbeats = beats_tempo[video_name]['vbeats']
    vbeats_per_bar = [[] for i in range(len(flow_magnitude_per_bar))]
    # tempo = beats_tempo[video_name]['tempo']
    tempo = 120   # 暂定2s一小节, 即120 bpm
    for vbeat in vbeats:
        start = int(vbeat['start'] * tempo / 60 * RESOLUTION / 4)  # absolute time to 1/16 beat
        vbeats_per_bar[start // RESOLUTION].append(start % RESOLUTION)

    l = [[0] * N_ENCODER_DIMENSION, [0] * N_ENCODER_DIMENSION]
    l[0][ENCODER_DIMENSION['type']] = preset_event2word['type']['Global']
    l[0][ENCODER_DIMENSION['key/genre']] = preset_event2word['key/genre'][key]
    l[1][ENCODER_DIMENSION['type']] = preset_event2word['type']['Global']
    l[1][ENCODER_DIMENSION['key/genre']] = preset_event2word['key/genre'][genre]

    for i, flow_magnitude in enumerate(flow_magnitude_per_bar):
        flow_magnitude = _cal_density(flow_magnitude)
        bar = [[0] * N_ENCODER_DIMENSION]
        bar[0][ENCODER_DIMENSION['type']] = preset_event2word['type']['Bar']
        bar[0][ENCODER_DIMENSION['density']] = preset_event2word['density'][flow_magnitude + 1]
        for vbeat in vbeats_per_bar[i]:
            bar += [[0] * N_ENCODER_DIMENSION]
            bar[-1][ENCODER_DIMENSION['type']] = preset_event2word['type']['Note']
            bar[-1][ENCODER_DIMENSION['beat']] = preset_event2word['beat']['Beat_%d' % (vbeat)]
        l += bar
    
    assert len(l) < ENCODER_MAX_LEN
    mask = [1] * len(l) + [0] * (ENCODER_MAX_LEN - len(l))
    
    return np.asarray(l, dtype=int), np.asarray(mask, dtype=int)


if __name__ == '__main__':
    video_dir = 'video_360p'
    for video_name in os.listdir(video_dir):
        target_path = os.path.join(video_dir, 'npz', video_name.replace('.mp4', '.npz'))
        if '.mp4' in video_name and not os.path.exists(target_path):
            print('processing to save to %s' % target_path)
            encoder, encoder_mask = video2numpy(video_name, video_dir)
            np.savez(target_path, encoder=encoder, encoder_mask=encoder_mask)
