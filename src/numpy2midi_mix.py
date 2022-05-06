import muspy
import numpy as np

from midi2numpy_mix import Note, DECODER_DIMENSION, RESOLUTION
from dictionary_mix import preset_event2word

INSTRUMENT_PROGRAM = {
    'Drums'  : 114,
    'Piano'  : 0,  # Acoustic Grand Piano
    'Guitar' : 24,  # Acoustic Guitar (nylon)
    'Bass'   : 33,  # Electric Bass (finger)
    'Strings': 41  # Viola
}


def test_numpy2midi(idx: int) -> muspy.Music:
    npz = np.load('lpd_5_ccdepr_mix_v4_10000.npz', allow_pickle=True)
    decoder = npz['x'][idx]
    name = npz['metadata'][idx]['id']
    return numpy2midi(name, decoder)


def numpy2midi(name, decoder: np.ndarray) -> muspy.Music:
    muspy_tracks = []
    # Decoder
    n_bars = -1
    beat = 0
    track_notes = {instr_type: [] for instr_type in INSTRUMENT_PROGRAM.keys()}
    for word in decoder:
        w_type = word[DECODER_DIMENSION['type']]
        if w_type == preset_event2word['type']['M']:
            if word[DECODER_DIMENSION['beat']] == preset_event2word['beat']['Bar']:
                n_bars += 1
            elif word[DECODER_DIMENSION['beat']] > 0 and word[DECODER_DIMENSION['beat']] < 17:
                beat = word[DECODER_DIMENSION['beat']] - 1
        elif w_type == preset_event2word['type']['Note']:
            note = Note()
            muspy_note = note.from_decoder_array(np_array=word, bar=n_bars, beat=beat)
            track_notes[note.instr_type].append(muspy_note)
        else:
            assert w_type == preset_event2word['type']['EOS']
            break
    for instr_type, muspy_notes in track_notes.items():
        muspy_tracks.append(muspy.Track(
            program=INSTRUMENT_PROGRAM[instr_type],
            is_drum=(instr_type == 'Drums'),
            name=instr_type,
            notes=muspy_notes
        ))

    muspy_music = muspy.Music(resolution=RESOLUTION // 4, tracks=muspy_tracks)

    muspy.write_midi(name + ".mid", muspy_music)

    return muspy_music


if __name__ == '__main__':
    test_numpy2midi(idx=66)
