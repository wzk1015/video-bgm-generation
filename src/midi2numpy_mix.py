import os
import math
import json

import muspy
import numpy as np
from tqdm import tqdm
import argparse

from dictionary_mix import preset_event2word, preset_word2event

#np.random.seed(208)

RESOLUTION = 16  # 每小节16个时间单位
DECODER_MAX_LEN = 10000
# DECODER_MAX_LEN = 3000
DECODER_DIMENSION = {
    'type'      : 0,
    'beat'      : 1,
    'density'   : 2,
    'pitch'     : 3,
    'duration'  : 4,
    'instr_type': 5,
    'strength'  : 6,
    'i_beat'    : 7,
    'n_beat'    : 8,
    'p_beat'    : 9,
}
N_DECODER_DIMENSION = len(DECODER_DIMENSION)
KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g',
        'g#', 'a', 'a#', 'b']


class Note:
    def __init__(self, muspy_note=None, instr_type=None):  # bar starts from 0
        if muspy_note is not None and instr_type is not None:
            self.time = muspy_note.time
            self.bar = self.time // RESOLUTION  # 从0开始
            self.beat = muspy_note.time % RESOLUTION  # 小节内部的第几个beat
            self.i_beat = self.bar * RESOLUTION + self.beat  # 整首歌的第几个beat
            self.pitch = muspy_note.pitch
            self.duration = min(RESOLUTION, muspy_note.duration)  # TODO: 截断过长的note?
            self.instr_type = instr_type
            # self.velocity = muspy_note.velocity
            self.velocity = 80

    def to_decoder_list(self, n_beat: int) -> list:
        l = [0] * N_DECODER_DIMENSION
        l[DECODER_DIMENSION['type']] = preset_event2word['type']['Note']
        l[DECODER_DIMENSION['pitch']] = preset_event2word['pitch']['Note_Pitch_%d' % self.pitch]
        l[DECODER_DIMENSION['duration']] = preset_event2word['duration']['Note_Duration_%d' % (self.duration * 120)]
        l[DECODER_DIMENSION['instr_type']] = preset_event2word['instr_type'][self.instr_type]
        l[DECODER_DIMENSION['i_beat']] = self.i_beat
        l[DECODER_DIMENSION['n_beat']] = n_beat
        l[DECODER_DIMENSION['p_beat']] = int(self.i_beat / n_beat * 100)
        return l

    def to_muspy_note(self) -> muspy.Note:
        return muspy.Note(time=self.time, pitch=self.pitch, duration=self.duration, velocity=self.velocity)

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
    def __init__(self, notes, i_bar):
        self.notes = notes
        self.n_notes = len(self.notes)
        self.i_bar = i_bar

    def _get_beat_token(self, note, density, strength, n_beat: int) -> list:
        l = [[0] * N_DECODER_DIMENSION]
        l[0][DECODER_DIMENSION['type']] = preset_event2word['type']['M']
        l[0][DECODER_DIMENSION['beat']] = preset_event2word['beat']['Beat_%d' % note.beat]
        l[0][DECODER_DIMENSION['density']] = density + 1
        l[0][DECODER_DIMENSION['instr_type']] = preset_event2word['instr_type'][note.instr_type]
        l[0][DECODER_DIMENSION['strength']] = strength
        l[0][DECODER_DIMENSION['i_beat']] = note.i_beat
        l[0][DECODER_DIMENSION['n_beat']] = n_beat
        l[0][DECODER_DIMENSION['p_beat']] = int(note.i_beat / n_beat * 100)
        return l

    def _get_bar_token(self, density, n_beat: int) -> list:
        l = [[0] * N_DECODER_DIMENSION]
        l[0][DECODER_DIMENSION['type']] = preset_event2word['type']['M']
        l[0][DECODER_DIMENSION['beat']] = preset_event2word['beat']['Bar']
        l[0][DECODER_DIMENSION['density']] = density + 1
        l[0][DECODER_DIMENSION['i_beat']] = self.i_bar * RESOLUTION
        l[0][DECODER_DIMENSION['n_beat']] = n_beat
        l[0][DECODER_DIMENSION['p_beat']] = int(self.i_bar * RESOLUTION / n_beat * 100)
        return l

    def to_decoder_list(self, n_beat: int) -> list:
        # 逆序构建
        n_beats = 0
        l = []
        if len(self.notes) > 0:
            # add the last note token
            l = [self.notes[-1].to_decoder_list(n_beat)]
            prev_note = self.notes[-1]
            n_notes_per_beat = 1
            for note in reversed(self.notes[:-1]):
                if note.beat != prev_note.beat or note.instr_type != prev_note.instr_type:
                    # add beat token
                    l = self._get_beat_token(note=prev_note, density=n_beats, strength=n_notes_per_beat,
                                             n_beat=n_beat) + l
                    if note.beat != prev_note.beat:
                        n_beats += 1
                    n_notes_per_beat = 0
                # add note token
                l = [note.to_decoder_list(n_beat)] + l
                n_notes_per_beat += 1
                prev_note = note
            # add the first beat token
            l = self._get_beat_token(note=prev_note, density=n_beats, strength=n_notes_per_beat, n_beat=n_beat) + l
            n_beats += 1
        # add bar token
        l = self._get_bar_token(density=n_beats, n_beat=n_beat) + l
        return l


class MIDI:
    def __init__(self, id: str):
        self.id = id
        self.midi = muspy.read_midi(os.path.join(midi_dir, id + '.mid'))
        self.midi.adjust_resolution(target=RESOLUTION // 4)

        self.n_beat = self.midi.get_end_time()
        self.n_bars = math.ceil((self.n_beat + 1) / RESOLUTION)

        muspy_tracks = []
        for track in self.midi.tracks:  # filter tracks with <=20 notes
            if len(track.notes) > 20:
                muspy_tracks.append(track)
        self.midi.tracks = muspy_tracks
        self.instruments = [track.name for track in muspy_tracks]
        self.bars = self._get_bars()

    def _get_bars(self):
        bars = [[] for i in range(self.n_bars)]
        for i, track in enumerate(self.midi.tracks):
            for j, muspy_note in enumerate(track.notes):
                note = Note(muspy_note, track.name)
                bars[note.bar].append(note)
        new_bars = []
        for i in range(len(bars)):
            bars[i].sort(key=lambda x: x.time)
            new_bars.append(Bar(notes=bars[i], i_bar=i))
        return new_bars

    def to_decoder_list(self) -> list:
        l = []
        for bar in self.bars:
            l += bar.to_decoder_list(self.n_beat)
        l += [[0] * N_DECODER_DIMENSION]  # 多一个EOS
        mask = [1] * len(l)
        return l, mask, len(l)


def midi2numpy(id_list: list):
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)

    decoder = []
    decoder_mask = []
    metadata_list = []
    decoder_len = []

    for id in tqdm(id_list):
        id_filename = os.path.join(json_dir, id + '.json')
        if os.path.exists(id_filename):
            with open(id_filename, 'r') as f:
                load_dict = json.load(f)
                decoder_list = load_dict['decoder_list']
                de_mask = load_dict['de_mask']
                de_len = load_dict['de_len']
                decoder_len.append(de_len)
                metadata = load_dict['metadata']
        else:

            midi = MIDI(id)
            decoder_list, de_mask, de_len = midi.to_decoder_list()

            decoder_len.append(de_len)
            if de_len > DECODER_MAX_LEN:
                continue

            # Padding to MAX_LEN
            decoder_list += [[0] * N_DECODER_DIMENSION] * (DECODER_MAX_LEN - de_len)
            de_mask += [0] * (DECODER_MAX_LEN - de_len)

            metadata = {'id': id, 'de_len': de_len, 'instruments': midi.instruments, 'genre': "N/A"}
            ##### genre set to empty

            dic = {'decoder_list': decoder_list, 'de_mask': de_mask, 'de_len': de_len, 'metadata': metadata}
            with open(id_filename, 'w') as f:
                json.dump(dic, f)

        decoder.append(decoder_list)
        decoder_mask.append(de_mask)
        metadata_list.append(metadata)

    print('max decoder length: %d' % max(decoder_len))

    decoder = np.asarray(decoder, dtype=int)
    x = decoder[:, :-1]
    y = decoder[:, 1:]
    decoder_mask = np.asarray(decoder_mask, dtype=int)

    np.savez(npz_filename, x=x, y=y, decoder_mask=decoder_mask, metadata=metadata_list)
    print(npz_filename)
    print('%d songs' % len(decoder))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_dir", default="../../lpd_5_cleansed_midi/", required=True)
    parser.add_argument("--out_name", default="data.npz", required=True)
    args = parser.parse_args()

    midi_dir = args.midi_dir
    npz_filename = os.path.join("../dataset/", args.out_name)
    json_dir = os.path.join("../dataset/json/")

    id_list = []
    for name in os.listdir(midi_dir):
        if name.endswith(".mid"):
            id_list.append(name[:-4])
        elif name.endswith(".midi"):
            id_list.append(name[:-5])
    
    midi2numpy(id_list)
