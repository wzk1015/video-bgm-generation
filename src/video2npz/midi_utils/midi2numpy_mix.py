import os
import math
import json
import pdb
import sys
import muspy
import h5py
import numpy as np
from tqdm import tqdm

from dictionary_mix import preset_event2word, preset_word2event


#np.random.seed(208)

RESOLUTION = 16  # 每小节16个时间单位
ENCODER_MAX_LEN = 1500
DECODER_MAX_LEN = 10000
ENCODER_DIMENSION = {
    'beat':         0,
    'b_dens':       1,
    'o_dens':       2,
}
N_ENCODER_DIMENSION = len(ENCODER_DIMENSION)
DECODER_DIMENSION = {
    'type':         0,
    'beat':         1,
    'b_dens':       2,
    'pitch':        3,
    'duration':     4,
    'instr_type':   5,
    'o_dens':       6,
}
N_DECODER_DIMENSION = len(DECODER_DIMENSION)
KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']

ONSET_DENSITY_QUANTILE = [33, 51, 65, 77, 89, 101, 116, 134, 162, 586]