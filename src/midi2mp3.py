import note_seq
from pretty_midi import PrettyMIDI
import midi2audio
import argparse

SAMPLE_RATE = 16000
SF2_PATH = '../SGM-v2.01-Sal-Guit-Bass-V1.3.sf2'


def midi_to_mp3(midi_path, tempo, mp3_path):
    midi_obj = PrettyMIDI(midi_path)
    # convert tempo
    midi_length = midi_obj.get_end_time()
    midi_obj.adjust_times([0, midi_length], [0, midi_length*120/tempo])
    processed_mid = midi_path[:-4] + "_processed.mid"
    midi_obj.write(processed_mid)

    print("converting into mp3")
    fs = midi2audio.FluidSynth(SF2_PATH, sample_rate=SAMPLE_RATE)
    fs.midi_to_audio(processed_mid, mp3_path)
    
    print("playing music")
    
    ns = note_seq.midi_io.midi_to_note_sequence(midi_obj)
    note_seq.play_sequence(ns, synth=note_seq.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
    note_seq.plot_sequence(ns)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../inference/get_0.mid")
    parser.add_argument("--tempo", default="96")
    parser.add_argument("--output", default="../inference/get_0.mp3")
    args = parser.parse_args()
    midi_to_mp3(args.input, float(args.tempo), args.output)
