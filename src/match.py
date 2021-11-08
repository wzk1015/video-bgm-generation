import argparse
from tqdm import tqdm
import numpy as np


def _get_density(bar_word):
    assert _is_bar_word(bar_word)
    if len(bar_word) == 10:
        return bar_word[2] - 1
    elif len(bar_word) == 6:
        return bar_word[1] - 1
    else:
        raise NotImplementedError


def _get_strength_and_tick(beat_word):
    assert _is_beat_word(beat_word)
    if len(beat_word) == 10:
        return beat_word[6], beat_word[1] - 1
    elif len(beat_word) == 6:
        return beat_word[2], beat_word[0] - 1
    else:
        raise NotImplementedError


def _is_bar_word(word):
    if len(word) == 10:
        return word[0] == 1 and word[1] == 17
    elif len(word) == 6:
        return word[0] == 17
    else:
        raise NotImplementedError


def _is_beat_word(word):
    if len(word) == 10:
        return word[0] == 1 and word[1] > 0 and word[1] < 17
    elif len(word) == 6:
        return word[0] > 0 and word[0] < 17
    else:
        raise NotImplementedError


def _get_density_and_strength_from_npz(npz):
    l_density = []
    l_strength = []
    for word in npz:
        if _is_bar_word(word):
            l_density.append(_get_density(word))
            l_strength.append([0] * 16)
        elif _is_beat_word(word):
            strength, tick = _get_strength_and_tick(word)
            l_strength[-1][tick] = strength
    return np.asarray(l_density), np.asarray(l_strength)


def cal_matchness(midi_npz, v_density, v_strength):
    m_density, m_strength = _get_density_and_strength_from_npz(midi_npz)

    n_bar = min(v_density.shape[0], m_density.shape[0])
    v_density = v_density[:n_bar]
    v_strength = v_strength[:n_bar]
    m_density = m_density[:n_bar]
    m_strength = m_strength[:n_bar]

    m_strength *= (v_strength > 0)
    dist = ((v_density - m_density) ** 2).mean() + ((v_strength - m_strength) ** 2).mean()
    return 1. / dist


def match_midi(video_npz, all_midi_metadata, all_midi_npz):
    res = []
    v_density, v_strength = _get_density_and_strength_from_npz(video_npz)

    print('Computing matching scores:')
    for i, midi_npz in enumerate(tqdm(all_midi_npz)):
        matchess = cal_matchness(midi_npz, v_density, v_strength)
        res.append((all_midi_metadata[i]['id'], matchess))
    res.sort(key=lambda x: x[1], reverse=True)

    print('IDs and matching scores of the 5 most matching music pieces:')
    for i in range(5):
        print(res[i][0], res[i][1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', default='../inference/wzk.npz', help='Video npz path')
    parser.add_argument('music_lib', default='../dataset/lpd_5_prcem_mix_v8_10000.npz', help='Music npz path')
    args = parser.parse_args()

    video_npz = np.load(args.video, allow_pickle=True)['input']
    tmp = np.load(args.music_lib, allow_pickle=True)
    all_midi_metadata = tmp['metadata']
    all_midi_npz = tmp['x']
    del tmp
    match_midi(video_npz, all_midi_metadata, all_midi_npz)