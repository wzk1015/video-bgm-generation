# %load VisBeat/MidiParse.py
import mido
import numpy as np
import os
# from TimeSignal1D import *
from .Audio import *
from .Event import *

class VBMIDITrack(object):
    def __init__(self, track, ticks_per_beat):
        self.mido_track = track;
        self.name = track.name;
        # print('Track {}: {}'.format(i, track.name));
        self.tempo = next((msg.tempo for msg in track if msg.type == 'set_tempo'), None);
        self.ticks_per_beat = ticks_per_beat;
        # self._get_note_on_times();
        self.note_on_times = None;


    def getBPM(self):
        return mido.tempo2bpm(self.tempo);

    def getNoteOnTimes(self, include_negative=None):
        if(self.note_on_times is None or (include_negative)):
            self._get_note_on_times(include_negative = include_negative);
        return self.note_on_times;


    def _get_note_on_times(self, include_negative = None):
        """
        Gets the starting time of each note.
        """
        note_times = []
        current_time = 0;
        for msg in self.mido_track:
            if not msg.is_meta:
                delta_time = mido.tick2second(msg.time, self.ticks_per_beat, self.tempo)
                current_time += delta_time
                if msg.type == 'note_on' and msg.velocity > 0:
                    if (include_negative or current_time>=0):
                        note_times.append(current_time);


        self.note_on_times = np.array(note_times);


    def get_note_durations(self, track_num):
        note_durations = []
        currently_played_notes = {}
        current_time = 0
        for msg in self.mido_track:
            if not msg.is_meta:
                delta_time = mido.tick2second(msg.time, self.ticks_per_beat, self.tempo)
                current_time += delta_time
                if msg.type == 'note_on' and msg.velocity > 0 and msg.note not in currently_played_notes:
                    currently_played_notes[msg.note] = current_time
                    #if len(currently_played_notes) > 1:
                    #    print "Number of played notes have reached: ", len(currently_played_notes)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    duration = current_time - currently_played_notes[msg.note]
                    note_durations.append((currently_played_notes[msg.note], duration))
                    currently_played_notes.pop(msg.note)
        assert not bool(currently_played_notes), "Finished looping through all messages. There should not be any notes playing at this point."
        #print "before sorting: ", note_durations
        note_durations.sort(key=lambda x : x[0])
        #print "after sorting: ", note_durations
        return np.array([ x[1] for x in note_durations ])

    def get_mouth_events(self):
        # note_times = []
        current_time = 0;
        number_on = 0;
        events = [];
        open_buffer = 0.1;
        close_buffer = 0.1
        last_type = 0;
        events.append(Event(start=0, type='mouth_close', weight=1));
        for msg in self.mido_track:
            if not msg.is_meta:
                delta_time = mido.tick2second(msg.time, self.ticks_per_beat, self.tempo)
                current_time += delta_time
                last_time = events[-1].start;
                passed = current_time-last_time;

                if msg.type == 'note_on' and msg.velocity > 0:
                    if(last_type==0):
                        if(passed>open_buffer):
                            events.append(Event(start=current_time-open_buffer, type='mouth_closed', weight=0));
                    events.append(Event(start=current_time, type='mouth_open', weight = truediv(msg.velocity,127)));
                    number_on = number_on + 1;
                    last_type=1;
                if(msg.type == 'note_off' or (msg.type=='note_on' and msg.velocity == 0)):
                    if (last_type == 1):
                        if (passed > close_buffer):
                            events.append(Event(start=current_time - close_buffer, type='mouth_opened', weight=0));
                    events.append(Event(start=current_time, type='mouth_close', weight = truediv(msg.velocity,127)));
                    number_on = number_on-1;
                    last_type=0;
                    # assert(number_on>-1);
                    # if(number_on==0):
        return events;
        # self.note_on_times = np.array(note_times);

    def getNoteOnTimesAsAudio(self, sampling_rate = None, note_sound=None, n_seconds=None):
        assert(n_seconds is not None), "must provide n seconds"
        if(sampling_rate is None):
            sampling_rate = 16000;
        if(note_sound is None):
            note_sound = Audio.getPing();
        s = Audio.Silence(n_seconds=n_seconds, sampling_rate=sampling_rate, name=self.name)
        s = s.getWithSoundAdded(note_sound, self.getNoteOnTimes());
        return s;



class VBMIDI(TimeSignal1D):
    def __init__(self, path=None):
        TimeSignal1D.__init__(self, path=path);
        self.midi_file = mido.MidiFile(path)
        self.tracks = [];
        for i, track in enumerate(self.midi_file.tracks):
            self.tracks.append(VBMIDITrack(track, self.midi_file.ticks_per_beat));

    def getNoteOnTimes(self, include_negative=None):
        return self.tracks[-1].getNoteOnTimes(include_negative=include_negative);

    def getMouthEvents(self):
        return self.tracks[-1].get_mouth_events();

    def getNoteOnTimesAsAudio(self, sampling_rate=None, note_sound=None):
        s = self.tracks[-1].getNoteOnTimesAsAudio(sampling_rate = sampling_rate, note_sound = note_sound, n_seconds = self.midi_file.length);
        if (s.name is None):
            s.name = self.getInfo('file_name')
        return s;