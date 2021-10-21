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
# DECODER_MAX_LEN = 3000
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
    # 'velocity': 4
}
N_DECODER_DIMENSION = len(DECODER_DIMENSION)
KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b']

ONSET_DENSITY_QUANTILE = [33, 51, 65, 77, 89, 101, 116, 134, 162, 586]

# with open('/mnt/data3/brd/music_dataset/LPD/lpd_5/lastfm.json', 'r') as f:
#     genre_dict = json.load(f)
#     i2g = genre_dict['i2g']
#     '''
#     Genre   Num     Percentage
#     --------------------------
#     Happy   1102    54.02%
#     Sad     938     45.98%
#     '''

# 更新genre时，记得更新dictionary
with open('/mnt/data3/brd/music_dataset/LPD/lpd_5/lastfm3.json', 'r') as f:
    genre_dict = json.load(f)
    i2g = genre_dict['i2g']
    '''
    Genre       Num     Percentage
    --------------------------
    classic	    1660	19.30%
    country	    799 	9.29%
    dance	    2147	24.96%
    electronic	596 	6.93%
    pop	        2368	27.53%
    rock	    1031	11.99%
    --------------------------
    8601 songs, 6 genres
    '''

midi_dir = '/mnt/data3/brd/music_dataset/LPD/lpd_5/lpd_5_cleansed_midi/'
h5_dir = '/mnt/data3/brd/music_dataset/LPD/lmd_matched_h5/'
npz_dir = '/mnt/data3/brd/music_dataset/LPD/lpd_5/lpd_5_cleansed_npz/'


class Note:
    def __init__(self, muspy_note=None, instr_type=None, sample=False):  # bar starts from 0
        if muspy_note is not None and instr_type is not None:
            self.time = muspy_note.time
            self.bar = self.time // RESOLUTION  # 从0开始
            self.beat = muspy_note.time % RESOLUTION
            self.pitch = muspy_note.pitch
            self.duration = min(RESOLUTION, muspy_note.duration)  # TODO: 截断过长的note?
            self.instr_type = instr_type
            # self.velocity = muspy_note.velocity
            self.sample = sample
            self.velocity = 80

    def to_encoder_list(self) -> list:
        l = [0] * N_ENCODER_DIMENSION
        l[ENCODER_DIMENSION['beat']] = preset_event2word['beat']['Beat_%d' % self.beat]
        return l

    def to_decoder_list(self) -> list:
        l = [0] * N_DECODER_DIMENSION
        l[DECODER_DIMENSION['type']] = preset_event2word['type']['Note']
        l[DECODER_DIMENSION['pitch']] = preset_event2word['pitch']['Note_Pitch_%d' % self.pitch]
        l[DECODER_DIMENSION['duration']] = preset_event2word['duration']['Note_Duration_%d' % (self.duration * 120)]
        l[DECODER_DIMENSION['instr_type']] = preset_event2word['instr_type'][self.instr_type]
        return l

    def to_muspy_note(self) -> muspy.Note:
        return muspy.Note(time=self.time, pitch=self.pitch, duration=self.duration, velocity=self.velocity)

    def from_encoder_array(self, np_array: np.ndarray, bar: int) -> muspy.Note:
        assert np_array[ENCODER_DIMENSION['beat']] > 0 and np_array[ENCODER_DIMENSION['beat']] < 17
        self.bar = bar
        self.beat = np_array[ENCODER_DIMENSION['beat']] - 1
        self.time = bar * RESOLUTION + self.beat
        self.pitch = 40
        self.duration = 0
        self.velocity = 100
        return self.to_muspy_note()
    
    # @staticmethod
    def from_decoder_array(self, np_array: np.ndarray, bar: int, beat: int) -> muspy.Note:
        assert np_array[DECODER_DIMENSION['type']] == preset_event2word['type']['Note']
        assert np_array[DECODER_DIMENSION['pitch']] > 0
        assert np_array[DECODER_DIMENSION['duration']] > 0
        assert np_array[DECODER_DIMENSION['instr_type']] > 0
        self.time = bar * RESOLUTION + beat
        self.pitch = np_array[DECODER_DIMENSION['pitch']] - 1
        self.duration = np_array[DECODER_DIMENSION['duration']] - 1
        self.instr_type = preset_word2event['instr_type'][np_array[DECODER_DIMENSION['instr_type']]]
        self.velocity = 80
        return self.to_muspy_note()


class Bar:
    def __init__(self, notes):
        self.notes = notes
        self.n_notes = len(self.notes)
        self.sampled_notes = self._get_sampled_notes()
        self.density = self._cal_onset_density()

    def _get_sampled_notes(self):
        sampled_notes = []
        for note in self.notes:
            if note.sample and (sampled_notes == [] or sampled_notes[-1].beat < note.beat):
                sampled_notes.append(note)
        return sampled_notes

    def _cal_onset_density(self):
        for i, quantile in enumerate(ONSET_DENSITY_QUANTILE):
            if self.n_notes < quantile:
                return i
        return len(ONSET_DENSITY_QUANTILE)

    # def _cal_pitch_variance(self):
    #     # TODO: _cal_pitch_variance
    #     return 1

    def to_encoder_list(self) -> list:
        l = [[0] * N_ENCODER_DIMENSION]
        l[0][ENCODER_DIMENSION['beat']] = preset_event2word['beat']['Bar']
        l[0][ENCODER_DIMENSION['b_dens']] = self.density + 1

        for note in self.sampled_notes:
            l.append(note.to_encoder_list())
        return l

    def to_decoder_list(self) -> list:
        l = [[0] * N_DECODER_DIMENSION]
        l[0][DECODER_DIMENSION['type']] = preset_event2word['type']['M']
        l[0][DECODER_DIMENSION['beat']] = preset_event2word['beat']['Bar']
        l[0][DECODER_DIMENSION['b_dens']] = self.density + 1

        i_last_beat = 0
        n_note_per_beat = 0
        for i, note in enumerate(self.notes):
            if i == 0: # first beat
                l += [[0] * N_DECODER_DIMENSION]
                l[-1][DECODER_DIMENSION['type']] = preset_event2word['type']['M']
                l[-1][DECODER_DIMENSION['beat']] = preset_event2word['beat']['Beat_%d' % note.beat]
                l[-1][DECODER_DIMENSION['instr_type']] = preset_event2word['instr_type'][note.instr_type]
                i_last_beat = len(l) - 1
            elif note.beat != self.notes[i-1].beat or note.instr_type != self.notes[i-1].instr_type:  # another beat
                l += [[0] * N_DECODER_DIMENSION]
                l[-1][DECODER_DIMENSION['type']] = preset_event2word['type']['M']
                l[-1][DECODER_DIMENSION['beat']] = preset_event2word['beat']['Beat_%d' % note.beat]
                l[-1][DECODER_DIMENSION['instr_type']] = preset_event2word['instr_type'][note.instr_type]
                
                # update o_dens for the previous beat token
                l[i_last_beat][DECODER_DIMENSION['o_dens']] = n_note_per_beat + 1  # 加1，因为None==0
                n_note_per_beat = 0
                i_last_beat = len(l) - 1
            l.append(note.to_decoder_list())
            n_note_per_beat += 1
        if n_note_per_beat > 0:  # update o_dens for the last beat token
            l[i_last_beat][DECODER_DIMENSION['o_dens']] = n_note_per_beat + 1  # 加1，因为None==0
        return l


class MIDI:
    def __init__(self, id: str):
        self.id = id
        self.midi = muspy.read_midi(os.path.join(midi_dir, id + '.mid'))
        self.midi.adjust_resolution(target=RESOLUTION//4)
        self.h5 = h5py.File(os.path.join(h5_dir, id + '.h5'), 'r')

        self.n_bars = math.ceil((self.midi.get_end_time() + 1) / RESOLUTION)
        self.genre = self._get_genre()
        self.key = self._get_key()
        # self.tempo = self._get_tempo()

          # filter tracks with <=20 notes
        muspy_tracks = []
        for track in self.midi.tracks:
            if len(track.notes) > 20:
                muspy_tracks.append(track)
        self.midi.tracks = muspy_tracks
        # self.n_drum_notes = 0 if self.midi.tracks[0].name != 'Drums' else len(self.midi.tracks[0].notes)

        # 计算sample概率，只从第一个track中抽
        n_notes = len(self.midi.tracks[0].notes)
        sample_prob = self.n_bars / n_notes  # 第一个track平均每个bar sample一个
        self.sample_flag = np.random.random(n_notes) < sample_prob

        self.bars = self._get_bars()

    def _get_genre(self):
        return i2g[self.id]

    def _get_key(self):
        # https://developer.spotify.com/documentation/web-api/reference/#endpoint-get-audio-features
        key = int(self.h5['analysis']['songs']['key'])  # 12种: 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'
        mode = int(self.h5['analysis']['songs']['mode'])  # 2种: 0: Minor, 1: Major
        return KEYS[12 * (1 - mode) + key]

    # def _get_tempo(self):
    #     return self.midi.tempos[0]

    def _get_bars(self):
        bars = [[] for i in range(self.n_bars)]
        for i, track in enumerate(self.midi.tracks):
            for j, muspy_note in enumerate(track.notes):
                sample = ((i == 0) and self.sample_flag[j])  # 只从第一个track sample
                note = Note(muspy_note, track.name, sample)
                bars[note.bar].append(note)
        new_bars = []
        for i in range(len(bars)):
            bars[i].sort(key=lambda x: x.time)
            new_bars.append(Bar(bars[i]))
        return new_bars

    def to_encoder_list(self) -> list:
        l = []
        for bar in self.bars:
            l += bar.to_encoder_list()
        mask = [1] * len(l)
        return l, mask, len(l)

    def to_decoder_list(self) -> list:
        l = []
        for bar in self.bars:
            l += bar.to_decoder_list()
        l += [[0] * N_DECODER_DIMENSION]  # 多一个EOS
        mask = [1] * len(l)
        return l, mask, len(l)


def midi2numpy(id_list: list):
    npz_filename = 'lpd_5_ccdepr_mix_v3_%d.npz' % DECODER_MAX_LEN

    encoder = []
    decoder = []
    encoder_mask = []
    decoder_mask = []
    metadata = []

    encoder_len = []
    decoder_len = []
        
    for id in tqdm(id_list):
        id_filename = os.path.join('/home/brd/projects/music_generation/compound-word-transformer/lpd_dataset/mix/json_mix_v2', id + '.json')
        if os.path.exists(id_filename):
            with open(id_filename, 'r') as f:
                load_dict = json.load(f)
                encoder_list = load_dict['encoder_list']
                decoder_list = load_dict['decoder_list']
                en_mask = load_dict['en_mask']
                de_mask = load_dict['de_mask']
                en_len = load_dict['en_len']
                de_len = load_dict['de_len']
        else:
            midi = MIDI(id)
            if midi.midi.tracks[0].name != 'Drums':  # 必须包含 Drums track
                continue
            encoder_list, en_mask, en_len = midi.to_encoder_list()
            decoder_list, de_mask, de_len = midi.to_decoder_list()
            
            encoder_len.append(en_len)
            decoder_len.append(de_len)

            if en_len > ENCODER_MAX_LEN or de_len > DECODER_MAX_LEN:
                continue

            # Padding to MAX_LEN
            encoder_list += [[0] * N_ENCODER_DIMENSION] * (ENCODER_MAX_LEN - en_len)
            decoder_list += [[0] * N_DECODER_DIMENSION] * (DECODER_MAX_LEN - de_len)
            en_mask += [0] * (ENCODER_MAX_LEN - en_len)
            de_mask += [0] * (DECODER_MAX_LEN - de_len)

            dic = {'encoder_list': encoder_list, 'decoder_list': decoder_list, 'en_mask': en_mask, 'de_mask': de_mask, 'en_len': en_len, 'de_len': de_len}
            with open(id_filename, 'w') as f:
                json.dump(dic, f)

        encoder.append(encoder_list)
        decoder.append(decoder_list)
        encoder_mask.append(en_mask)
        decoder_mask.append(de_mask)
        metadata.append({'id': id, 'en_len': en_len, 'de_len': de_len})

    print(max(encoder_len), max(decoder_len))
    import matplotlib.pyplot as plt
    plt.hist(decoder_len)
    plt.hist(encoder_len)
    plt.savefig('len.jpg')

    encoder = np.asarray(encoder, dtype=int)
    decoder = np.asarray(decoder, dtype=int)
    x = decoder[:, :-1]
    y = decoder[:, 1:]
    encoder_mask = np.asarray(encoder_mask, dtype=int)
    decoder_mask = np.asarray(decoder_mask, dtype=int)

    np.savez(npz_filename, encoder=encoder, x=x, y=y, encoder_mask=encoder_mask, decoder_mask=decoder_mask, metadata=metadata)
    print(npz_filename)
    print('%d songs' % len(encoder))


if __name__ == '__main__':
    id_list = list(i2g.keys())
    midi2numpy(id_list[:])
