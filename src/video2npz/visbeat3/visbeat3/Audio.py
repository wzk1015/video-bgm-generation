from .EventList import *
from .TimeSignal1D import *
from scipy.io.wavfile import write
import librosa
import librosa.display

from scipy import signal, fftpack



class Audio(TimeSignal1D):
    """Audio (class): A sound, and a bunch of convenience functions to go with it.
        Attributes:
            x: the sound signal
            sampling_rate: the sampling rate
    """

    FEATURE_FUNCS = TimeSignal1D.FEATURE_FUNCS.copy();

    def __init__(self, path=None, sampling_rate=None, x=None, name=None):
        #VBObject.__init__(self, path=path);
        TimeSignal1D.__init__(self, path=path, sampling_rate=sampling_rate, x=x);
        #self.initializeBlank();
        if(name is not None):
            self.name = name;
        if(path):
            self.loadFile();
        if(self.name is None):
            self.name = self.getInfo('file_name')


    # <editor-fold desc="Property: 'name'">
    @property
    def name(self):
        return self.getName();
    def getName(self):
        return self._name;
    @name.setter
    def name(self, value):
        self._setName(value);
    def _setName(self, value):
        self._name = value;
    # </editor-fold>

    def initializeBlank(self):
        self.name = None;
        TimeSignal1D.initializeBlank(self);
        self.n_channels = 1;

    def _getFrameRate(self):
        return self.getOnsetSamplingRate();

    def clone(self):
        clone = Audio();
        clone.setPath(self.getPath());
        clone.x = self.x.copy();
        clone.sampling_rate = self.sampling_rate;
        clone.n_channels = self.n_channels;
        stereo = self.getStereo();
        if (stereo is not None):
            clone.setInfo('stereo_signal', stereo);
            clone.setInfo('stereo_sampling_rate', self.getInfo('stereo_sampling_rate'));
        return clone;

    def loadFile(self, file_path=None, sampling_rate=None, convert_to_mono=True):
        if(file_path):
            self.setPath(file_path=file_path);

        if('file_path' in self.a_info):
            self.x, self.sampling_rate = librosa.load(self.a_info['file_path'],sr=sampling_rate, mono=convert_to_mono);
            if(len(self.x.shape)>1):
                self.a_info['stereo_signal']=self.x;
                self.setInfo('stereo_sampling_rate', self.sampling_rate);
                self.x = np.mean(self.x, axis = 0)
                print("averaged stereo channels");

    def getStereo(self):
        return self.a_info.get('stereo_signal');

    def getStereoSamplingRate(self):
        return self.getInfo('stereo_sampling_rate');



    def getStringForHTMLStreamingBase64(self):
        encoded = self.getStereoEncodedBase64WAV();
        return "data:audio/wav;base64,{0}".format(encoded.decode('ascii'));

    def getStereoEncodedBase64WAV(self):
        sig = self.getStereo();
        sr = self.getStereoSamplingRate();
        if (sig is None):
            sig = self.getSignal();
            sr = self.sampling_rate;
        saudio = _make_wav(sig, sr);
        encoded = base64.b64encode(saudio);
        return encoded;

    def getMonoEncodedBase64WAV(self):
        sig = self.getSignal();
        sr = self.sampling_rate;
        saudio = _make_wav(sig, sr);
        encoded = base64.b64encode(saudio);
        return encoded;

    def getLocalRhythmicSaliency(self, **kwargs):
        return self.getOnsetEnvelope(**kwargs);


    def play(self, autoplay = None):
        """
         Play audio. Works in Jupyter. if notebook, audio will play normalized -- this is something html5 audio seems to do by default.
        :param autoplay:
        :return:
        """
        if(ISNOTEBOOK):
            # if(normalize):
            audiodisp = vb_get_ipython().display.Audio(data=self.getSignal(), rate=self.sampling_rate, autoplay=autoplay);
            vb_get_ipython().display.display(audiodisp);
            # vb_get_ipython().display.display(vb_get_ipython().display.Audio(data=self.getSignal(), rate=self.sampling_rate, autoplay=False))
            # else:
            # audiodisp = vb_get_ipython().display.Audio(data=self.getMonoEncodedBase64WAV(), rate=self.sampling_rate,
            #                                            autoplay=autoplay);
            # vb_get_ipython().display.display(audiodisp);
            # print("html render")
            # htmlstr = self.getStringForHTMLStreamingBase64()
            # ahtml = HTML(data='''<audio alt="{}" controls>
            #             <source src="{}" type="audio/wav" />
            #          </audio>'''.format(self.name, htmlstr));
            # IPython.display.display(ahtml);

        else:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                             channels=self.n_channels,
                             rate=self.sampling_rate,
                             output=True,
                             output_device_index=1
                             )
            stream.write(self.getSignal())
            stream.stop_stream();
            stream.close()
            p.terminate()

    def playBeats(self, indices=None, beats=None):
        beat_inds = indices;
        if(beats is None):
            beats = self.getBeatEvents();
        if(beat_inds is None):
            beat_inds = [0, len(beats)-1];
        if(not isinstance(beat_inds, list)):
            beat_inds=[beat_inds-1, beat_inds, beat_inds+1];
        if(beat_inds[0]<0):
            start_time = 0;
        else:
            start_time = beats[beat_inds[0]].start;
        if(beat_inds[-1]>len(beats)):
            end_time = self.getDuration();
        else:
            end_time = beats[beat_inds[-1]].start;
        self.playSegment(time_range = [start_time, end_time]);

    def AudioClipFromBeatRange(self, beat_range, beats=None):
        if(beats is None):
            beats = self.getBeatEvents();
        if(beat_range is None):
            beat_range = [0, len(beats)-1];
        if(beat_range[1] is None):
            beat_range[1]=len(beats)-1;

        return self.AudioClip(start=beats[beat_range[0]].start, end=beats[beat_range[1]].start);




    def playSegment(self, time_range, autoplay=None):
        start_time = time_range[0];
        end_time = time_range[1];
        if(isinstance(start_time, Event)):
            start_time = start_time.start;
        if(isinstance(end_time, Event)):
            end_time = end_time.start;
        if(ISNOTEBOOK):
            audiodisp = vb_get_ipython().display.Audio(data=self.getSignalSegment(time_range=[start_time, end_time]), rate=self.sampling_rate, autoplay=autoplay);
            vb_get_ipython().display.display(audiodisp);
            # vb_get_ipython().display.display(vb_get_ipython().display.Audio(data=self.getSignal(), rate=self.sampling_rate, autoplay=False))
        else:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paFloat32,
                             channels=self.n_channels,
                             rate=self.sampling_rate,
                             output=True,
                             output_device_index=1
                             )
            stream.write(self.getSignalSegment(time_range=[start_time, end_time]))
            stream.stop_stream();
            stream.close()
            p.terminate()

    def writeToFile(self, output_path=None, output_sampling_rate=None):
        assert(output_path), "must provide path to save audio."
        data = self.getSignal();
        scaled = np.int16(data/np.max(np.abs(data)) * 32767)
        if(output_sampling_rate is None):
            output_sampling_rate=44100
        write(output_path, output_sampling_rate, scaled)

    def setValueRange(self, value_range=None):
        if(value_range is None):
            value_range = [-1,1];
        TimeSignal1D.setValueRange(self, value_range=value_range);

    def resample(self, sampling_rate):
        new_n_samples = sampling_rate*self.getDuration();
        self.x = sp.signal.resample(self.x, int(new_n_samples));
        self.sampling_rate = sampling_rate;

    def GetResampled(self, sampling_rate):
        new_a = self.clone();
        new_a.resample(sampling_rate=sampling_rate);
        return new_a;



    def AlignedTo(self, B):
        assert(False),'Abe needs to fix -- broke when messing with alignment code for video';

        A = self.clone();
        if(A.sampling_rate !=B.sampling_rate):
            A.resample(B.sampling_rate);

        a_signal = A.getSignal();
        b_signal = B.getSignal();
        a_duration = self.getDuration();

        if(len(a_signal) < len(b_signal)):
            siglen = spotgt_shift_bit_length(len(b_signal));
        else:
            siglen = spotgt_shift_bit_length(len(a_signal));

        npada = siglen-len(a_signal);
        if(npada>0):
            a_signal = np.pad(a_signal, (0,npada), 'constant', constant_values=(0, 0));

        npadb = siglen-len(b_signal);
        if(npadb>0):
            b_signal = np.pad(b_signal, (0,npadb), 'constant', constant_values=(0, 0));

        Af = fftpack.fft(a_signal);
        Bf = fftpack.fft(b_signal);
        Ar = Af.conjugate();
        Br = Bf.conjugate();
        # Ar = -Af.conjugate();
        # Br = -Bf.conjugate();

        ashift = np.argmax(np.abs(fftpack.ifft(Ar * Bf)));
        print(ashift);
        print((np.argmax(np.abs(fftpack.ifft(Af * Br)))));

        a_return = Audio();
        a_return.n_channels = 1;
        a_return.sampling_rate = B.sampling_rate;
        a_return.x = np.roll(a_signal, ashift);

        return a_return;

    def getOffsetFrom(self, B):
        return -self.getShiftAmountTo(B);

    def getShiftAmountTo(self, B):
        """
        Get amount to shift by in seconds (to shift this to alignment with B)
        """
        a_signal = self.getSignal();
        b_signal = B.getSignal();
        a_duration = self.getDuration();
        if(self.sampling_rate !=B.sampling_rate):
            ansamps = a_duration*B.sampling_rate;
            a_signal = sp.signal.resample(a_signal, int(ansamps));
            # a_signal = librosa.resample(a_signal, self.sampling_rate, B.sampling_rate, res_type='kaiser_fast');
        if(len(a_signal) < len(b_signal)):
            siglen = spotgt_shift_bit_length(len(b_signal));
        else:
            siglen = spotgt_shift_bit_length(len(a_signal));
        npada = siglen-len(a_signal);
        if(npada>0):
            a_signal = np.pad(a_signal, (0,npada), 'constant', constant_values=(0, 0));
        npadb = siglen-len(b_signal);
        if(npadb>0):
            b_signal = np.pad(b_signal, (0,npadb), 'constant', constant_values=(0, 0));

        # if(len(a_signal) < len(b_signal)):
        #     npad = len(b_signal)-len(a_signal);
        #     a_signal = np.pad(a_signal, (0,npad), 'constant', constant_values=(0, 0));
        #     print('pad1');
        #
        # if (len(b_signal) < len(a_signal)):
        #     npad = len(a_signal) - len(b_signal);
        #     b_signal = np.pad(b_signal, (0, npad), 'constant', constant_values=(0, 0));
        #     print('pad2');

        # print('point0')

        Af = fftpack.fft(a_signal);
        Bf = fftpack.fft(b_signal);
        Ar = Af.conjugate();
        Br = Bf.conjugate();
        # Ar = -Af.conjugate();
        # Br = -Bf.conjugate();
        ashiftab = np.argmax(np.abs(fftpack.ifft(Ar * Bf)));
        durationsamples = self.getDuration()*B.sampling_rate;
        # if(ashiftab>(durationsamples*0.5)):
        #     ashiftab = ashiftab-durationsamples;
        return truediv(ashiftab, B.sampling_rate);

    def getBeatEventList(self, time_range = None):
        beats = self.getBeats();
        if(time_range is None):
            return EventList.FromStartTimes(beats, type='beats');

        start_beat = 0;
        end_beat = len(beats) - 1;
        if(time_range[0] is not None):
            while((start_beat<len(beats)) and (beats[start_beat]<time_range[0])):
                start_beat=start_beat+1;
        if(time_range[1] is not None):
            while(end_beat>0 and (beats[end_beat]>time_range[1])):
                end_beat = end_beat-1;
        if(end_beat>start_beat):
            return EventList.FromStartTimes(beats[start_beat:end_beat], type='beats');
        else:
            return None;


    def getBeatEvents(self, start_time=None, end_time=None):
        beat_eventlist = self.getBeatEventList();
        return beat_eventlist.events;

    def AudioClip(self, start, end):
        from . import AudioClip
        clip = AudioClip.AudioClip(path=self.getPath(), start=start, end=end);
        return clip;


    def getWithSoundAdded(self, add_times, sound=None, mute_original=None, gain_original = None):
        if(sound is None):
            sound = Audio.PingSound(sampling_rate=self.sampling_rate);
        if(sound.sampling_rate==self.sampling_rate):
            s_toadd = sound.getSignal();
        else:
            new_n_samples = self.sampling_rate * sound.getDuration();
            s_toadd = sp.signal.resample(sound.getSignal(), int(new_n_samples));

        result = self.clone();
        if(mute_original):
            result.x = np.zeros(result.x.shape);
        if(gain_original is not None):
            result.x = result.x*gain_original;


        sl = len(s_toadd);
        for t in add_times:
            if(t<self.getDuration()):
                ti = int(t*self.sampling_rate);
                te = min(len(result.x), ti+sl);
                result.x[ti:te]=result.x[ti:te]+s_toadd[0:te-ti];
        result.setValueRange();
        # result.setMaxAbsValue(1.0);
        return result;


    def showSpectrogram(self, time_range = None, **kwargs):
        if(hop_length is None):
            hop_length=AUDIO_DEFAULT_HOP_LENGTH;

        fig = plt.figure();

        S = self.getSpectrogram(hop_length=hop_length, **kwargs);
        hop_length = self.getFeatureParams('spectrogram').get('hop_length');
        print(("hop length: {}".format(hop_length)));
        # S = librosa.stft(self.getSignal(), hop_length=hop_length);

        librosa.display.specshow(librosa.amplitude_to_db(S,ref = np.max),sr=self.sampling_rate, hop_length=hop_length, y_axis = 'linear', x_axis = 'time')
        plt.title('Power Spectrogram')
        plt.colorbar(format='%+2.0f dB')
        plt.ylim([0,8000]);
        plt.xlabel('Time (s)')
        if (time_range is not None):
            plt.xlim(time_range);
        plt.tight_layout()
        return fig;

    def showMelSpectrogram(self, force_recompute=False, **kwargs):
        mel_spec = self.getMelSpectrogram(force_recompute=force_recompute, **kwargs);
        # Make a new figure
        plt.figure();
        plt.figure(figsize=(12,4))
        # Display the spectrogram on a mel scale
        # sample rate and hop length parameters are used to render the time axis
        librosa.display.specshow(mel_spec, sr=self.sampling_rate, x_axis='time', y_axis='mel')
        # Put a descriptive title on the plot
        plt.title('Mel Power Spectrogram')
        # draw a color bar
        plt.colorbar(format='%+02.0f dB')
        # Make the figure layout compact
        plt.tight_layout()


    @staticmethod
    def Silence(n_seconds, sampling_rate, name=None):
        x = np.zeros(int(np.ceil(n_seconds*sampling_rate)));
        s = Audio(x=x, sampling_rate = sampling_rate, name = name);
        if(s.name is None):
            s.name = 'silence';
        return s;


    @staticmethod
    def _getDampedSin(freq, n_seconds=None, sampling_rate=16000, damping=0.1, noise_floor=0.001):
        """

        :param freq:
        :param n_seconds: if None, will set length necessary to damp to noise floor, up to 10.0s.
        :param sampling_rate:
        :param damping: Every period the amplitude is multiplied by 0.5^damping
        :param noise_floor:
        :return:
        """
        if(n_seconds is None):
            if(damping>0.005):
                nosc = truediv(math.log(noise_floor,0.5),damping);
                n_seconds = truediv(nosc, freq);
            else:
                n_seconds = 10.0;

        x = np.sin(np.linspace(0, n_seconds * freq * 2 * np.pi, np.round(n_seconds * sampling_rate)));
        dmp = np.linspace(0, n_seconds * freq, np.round(n_seconds * sampling_rate))
        dmp = np.power(0.5, dmp * damping);
        return np.multiply(x, dmp);



    @staticmethod
    def PingSound(n_seconds=None, freqs=None, damping=None, sampling_rate = 16000):
        if(freqs is None):
            # freqs = [400,500,600, 700, 800, 900, 1000, 1100, 1200, 1300];
            # freqs = np.arange(4, 25) * 100
            freqs = np.arange(5, 25) * 75; # just kind of thought this sounded fine...
        if(damping is None):
            damping = [0.05]*len(freqs);
        if(not isinstance(damping,list)):
            damping = [damping]*len(freqs);
        s = Audio._getDampedSin(freq=freqs[0], n_seconds = n_seconds, sampling_rate = sampling_rate, damping=damping[0]);
        for h in range(1,len(freqs)):
            new_s = Audio._getDampedSin(freq=freqs[h], n_seconds = n_seconds, sampling_rate = sampling_rate, damping=damping[h]);
            if(len(new_s)>len(s)):
                new_s[:len(s)]=new_s[:len(s)]+s;
                s = new_s;
            else:
                s[:len(new_s)] = s[:len(new_s)]+new_s;
        sa = Audio(x=s, sampling_rate=sampling_rate, name = 'ping');
        # sa.setValueRange(value_range=[-1,1]);
        sa.setMaxAbsValue(1.0);
        return sa;


    ##### --- FEATURES --- #####

    def getBeats(self, use_full_signal=True, tightness = None, force_recompute=False):
        """

        :param use_full_signal: If called from AudioClip class, this will determine whether the full signal is used or
            just the clip. This is important because the tempo is a global property. More evidence for a constant tempo
            is good, but if the tempo changes over the audio you probably don't want to use the full signal.
        :param tightness: How tightly to the tempo are beats picked. Must be positive. 0 would not care about tempo.
            100 is default. This is actually part of the penalty used in the dynamic programming objective from
            Ellis 2007. In librosa: txwt = -tightness * (np.log(-window / period) ** 2)
        :param force_recompute:
        :return:
        """
        if ((not self.hasFeature(name='beats')) or force_recompute):
            beat_args = dict(sr = self.sampling_rate, units = 'time');

            if (use_full_signal):
                beat_args.update(dict(y = self.getFullSignal()));
            else:
                beat_args.update(dict(y=self.getSignal()))
            if (tightness is not None):
                beat_args.update(dict(tightness=tightness));

            # print(beat_args)
            tempo, beats = librosa.beat.beat_track(**beat_args);
            self.setFeature(name='tempo', value=tempo);
            self.setFeature(name='beats', value=beats);
        return self.getFeature(name='beats');

    def getBeatVector(self, vector_length=None, force_recompute=False):
        """use_full_signal only makes a difference in AudioClip subclass."""
        if ((not self.hasFeature(name='beatvector')) or force_recompute):
            D = self.getDuration();
            if (vector_length is None):
                vector_length = int(math.ceil(D * 240));
            rvec = np.zeros(vector_length);
            beats = self.getFeature('beats');
            step = truediv(D, vector_length);
            for i in range(len(beats)):
                b = beats[i];
                id = int(truediv(b, step));
                rvec[id] = 1;
            self.setFeature(name='beatvector', value=rvec);
        return self.getFeature(name='beatvector');

    def getOnsets(self, use_full_signal=True, force_recompute=False, **kwargs):
        """use_full_signal only makes a difference in AudioClip subclass."""
        if ((not self.hasFeature(name='onsets')) or force_recompute):
            if (use_full_signal):
                onsets = librosa.onset.onset_detect(y=self.getFullSignal(), sr=self.sampling_rate, units='time',
                                                    **kwargs);
                self.setFeature(name='onsets', value=onsets);
            else:
                onsets = librosa.onset.onset_detect(y=self.getSignal(), sr=self.sampling_rate, units='time',
                                                    **kwargs);
                self.setFeature(name='onsets', value=onsets);
        return self.getFeature(name='onsets');

    def getOnsetSamplingRate(self):
        return np.true_divide(self.sampling_rate, AUDIO_DEFAULT_HOP_LENGTH);

    def pickOnsets(self, pre_max_time=0.03,
                   post_max_time=0.0,
                   pre_avg_time=0.1,
                   post_avg_time=0.1,
                   wait_time=0.03,
                   delta=0.07,
                   force_recompute=True, **kwargs):
        """

        :param pre_max_time:
        :param post_max_time:
        :param pre_avg_time:
        :param post_avg_time:
        :param wait_time:
        :param force_recompute:
        :param kwargs:
        :return:
        """
        # kwargs.setdefault('pre_max', 0.03 * sr // hop_length)  # 30ms
        # kwargs.setdefault('post_max', 0.00 * sr // hop_length + 1)  # 0ms
        # kwargs.setdefault('pre_avg', 0.10 * sr // hop_length)  # 100ms
        # kwargs.setdefault('post_avg', 0.10 * sr // hop_length + 1)  # 100ms
        # kwargs.setdefault('wait', 0.03 * sr // hop_length)  # 30ms
        # kwargs.setdefault('delta', 0.07)

        pick_params = dict(
            pre_max_time=pre_max_time,
            post_max_time=post_max_time,
            pre_avg_time=pre_avg_time,
            post_avg_time=post_avg_time,
            wait_time=wait_time,
            delta=delta,
        )

        tp_keys = list(pick_params.keys());
        for p in tp_keys:
            pick_params[p] = int(round(self.getOnsetSamplingRate() * pick_params[p]));

        dparams = dict(
            pre_max=pick_params['pre_max_time'],
            post_max=pick_params['post_max_time'] + 1,
            pre_avg=pick_params['pre_avg_time'],
            post_avg=pick_params['post_avg_time'] + 1,
            wait=pick_params['wait_time'],
            delta=delta
        )

        return self.getOnsets(force_recompute=force_recompute, **dparams);

    def getEvents(self):
        return self.getBeatEvents();

    def getEventList(self):
        return EventList(self.getBeatEvents());

    def getOnsetEvents(self):
        onsets = self.getOnsets();
        events = Event.FromStartTimes(onsets, type='onsets');
        return events;

    def getBeatEvents(self, start_time=None, end_time=None, **kwargs):
        beats = self.getBeats(**kwargs);
        start_beat = 0;
        end_beat = len(beats) - 1;
        if(start_time is not None):
            while((start_beat<len(beats)) and (beats[start_beat]<start_time)):
                start_beat=start_beat+1;
        if(end_time is not None):
            while(end_beat>0 and (beats[end_beat]>end_time)):
                end_beat = end_beat-1;
        if(end_beat>start_beat):
            return Event.FromStartTimes(beats[start_beat:end_beat], type='beats');
        else:
            return None;

    def getOnsetEnvelope(self, use_full_signal=True, force_recompute=False, centering=True, **kwargs):
        """use_full_signal only makes a difference in AudioClip subclass."""
        feature_name = 'onset_envelope';
        if((not self.hasFeature(name=feature_name)) or force_recompute):
            if(use_full_signal):
                eval_sig = self.getFullSignal();
            else:
                eval_sig = self.getSignal();
            onsets = librosa.onset.onset_strength(y=eval_sig, sr=self.sampling_rate, centering=centering, hop_length=AUDIO_DEFAULT_HOP_LENGTH, **kwargs);

            self.setFeature(name=feature_name, value=onsets);
        return self.getFeature(name=feature_name);


    def getMelSpectrogram(self, n_mels = 128, force_recompute=False):
        feature_name = 'melspectrogram';
        if((not self.hasFeature(name=feature_name)) or force_recompute):
            params = dict( sr = self.sampling_rate,
                                n_mels = n_mels);
            Spec = librosa.feature.melspectrogram(self.getSignal(), **params);
            self.setFeature(name=feature_name, value=librosa.power_to_db(Spec, ref=np.max), params=params);
        return self.getFeature(feature_name);

    def getSpectrogram(self, hop_length=None, force_recompute=False, **kwargs):
        feature_name = 'spectrogram';
        if((not self.hasFeature(name=feature_name)) or force_recompute):
            # params = dict( sr = self.sampling_rate);
            # params.update(kwargs);
            params = dict(kwargs);

            if(hop_length is None):
                hop_length = AUDIO_DEFAULT_HOP_LENGTH;
            params['hop_length'] = hop_length;

            center = kwargs.get('center');
            if(center is None):
                center = True;
            params['center'] = center;
            # print('recomputing {}'.format(hop_length))

            S = np.abs(librosa.stft(self.getSignal(), **params));
            self.setFeature(name=feature_name, value=S, params=params);
        return self.getFeature(feature_name);

    def getRMSE(self, force_recompute=False, hop_length=None, frame_length=None):
        feature_name = 'rmse';
        if((not self.hasFeature(name=feature_name)) or force_recompute):
            if(frame_length is None):
                frac_of_second = 0.05;
                frame_length=max(1, int(self.sampling_rate*frac_of_second));
            if(hop_length is None):
                hop_length=int(math.floor(frame_length*0.5));
            params = dict( hop_length = hop_length,
                                center = True,
                                frame_length = frame_length);
            rmse = librosa.feature.rmse(y=self.getSignal(), **params);
            self.setFeature(name=feature_name, value=np.ndarray.flatten(rmse), params=params);
        return self.getFeature(feature_name);


    def getBeatTimeBefore(self, t):
        return self.getBeatBefore(t=t).start;

    def getBeatBefore(self, t):
        beats = self.getBeatEvents();
        bi = self.getBeatIndexBefore(t=t);
        return beats[bi].start;

    def getBeatIndexBefore(self, t):
        beats = self.getBeatEvents();
        for i, b in enumerate(beats):
            if(b.start>t):
                return i-1;
        return len(beats)-1;


    def getTempogram(self, window_length=None, force_recompute=None, frame_rate=None, resample_rate = None, **kwargs):
        """

        :param self:
        :param window_length: in seconds
        :param force_recompute:
        :param kwargs:
        :return:
        """
        feature_name = 'tempogram';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            if (window_length is None):
                window_length = DEFAULT_TEMPOGRAM_WINDOW_SECONDS;
            tpgparams = {};
            tpgparams.update(kwargs);

            sr = self.sampling_rate;
            y=self.getSignal();
            if(resample_rate is None):
                resample_rate = 22050;
            if(sr>resample_rate):
                print(("resampling {}Hz to {}Hz".format(sr, resample_rate)));
                y = librosa.core.resample(y, orig_sr=sr, target_sr=resample_rate);
                sr = resample_rate;
                print("resampled")

            if(frame_rate is None):
                frame_rate = 30;
            hop_length = int(round(truediv(sr,frame_rate)));
            win_length = int(round(window_length * frame_rate));

            tparams = dict(y=y,
                           sr=sr,
                           hop_length=hop_length,
                           win_length=win_length,
                           **kwargs);
            tpgparams.update(dict(sr=sr,hop_length=hop_length, win_length=win_length));
            result = librosa.feature.tempogram(**tparams);
            ###########
            tempo_bpms = librosa.tempo_frequencies(result.shape[0], hop_length=hop_length, sr=sr)
            self.setFeature(name='tempogram_bpms', value=tempo_bpms);
            self.setFeature(name=feature_name, value=result, params=tpgparams);
            self.setInfo(label='tempogram_params',value=tpgparams);

        return self.getFeature(feature_name);

    def plotTempogram(self, window=None, time_range=None, **kwargs):
        tempogram = self.getFeature('tempogram', force_recompute=True);
        tparams = self.getInfo('tempogram_params')
        toshow = tempogram;
        if(window is not None):
            wstart=int(round(window[0]*self.sampling_rate));
            wend = int(round(window[1]*self.sampling_rate));
            toshow = tempogram[:,wstart:wend];

        mplt = librosa.display.specshow(toshow, sr=tparams['sr'], hop_length=tparams['hop_length'], x_axis = 'time', y_axis = 'tempo')
        plt.legend(frameon=True, framealpha=0.75)
        plt.set_cmap('coolwarm')
        plt.colorbar(format='%+2.0f dB')
        if (time_range is not None):
            plt.xlim(time_range);
        plt.xlabel('Time (s)')
        plt.title('Audio Tempogram');
        plt.tight_layout()
        return mplt;

    def plotOnsets(self, **kwargs):
        signal = self.getFeature('onset_envelope');
        events = self.getOnsetEvents();
        mplt = Event.PlotSignalAndEvents(signal, sampling_rate=self.getOnsetSamplingRate(), events=events, **kwargs);
        plt.xlabel('Time (s)')
        plt.ylabel('Onset Strength')
        return mplt;

    def plotBeats(self, **kwargs):
        signal = self.getFeature('onset_envelope');
        events = self.getBeatEvents();
        mplt = Event.PlotSignalAndEvents(signal, sampling_rate=self.getOnsetSamplingRate(), events=events, **kwargs);
        plt.title('Onset Envelope and Beats');
        plt.xlabel('Time (s)')
        plt.ylabel('Onset Strength')
        return mplt;

    def plotOnsetEnvelope(self, **kwargs):
        signal = self.getFeature('onset_envelope');
        events = None;
        mplt = Event.PlotSignalAndEvents(signal, sampling_rate=self.getOnsetSamplingRate(), events=events, **kwargs);
        plt.title('Onset Envelope');
        plt.xlabel('Time (s)')
        plt.ylabel('Onset Strength')
        return mplt;

    def plotSignal(self, time_range=None, ylim=None, **kwargs):
        signal = self.getSignal();
        # Event.PlotSignalAndEvents(signal, sampling_rate=self.getOnsetSamplingRate(), events=events, **kwargs);
        times = np.arange(len(signal));
        times = times * truediv(1.0, self.sampling_rate);
        mplt = plt.plot(times, signal);
        if (time_range is not None):
            plt.xlim(time_range[0], time_range[1])
        if (ylim is not None):
            plt.ylim(ylim);
        plt.title('Time Signal');
        return mplt;


    FEATURE_FUNCS['melspectrogram'] = getMelSpectrogram;
    FEATURE_FUNCS['spectrogram'] = getSpectrogram;
    FEATURE_FUNCS['rmse'] = getRMSE;
    FEATURE_FUNCS['beats'] = getBeats;
    FEATURE_FUNCS['onsets'] = getOnsets;
    FEATURE_FUNCS['beatvector'] = getBeatVector;
    FEATURE_FUNCS['tempogram'] = getTempogram;
    FEATURE_FUNCS['onset_envelope'] = getOnsetEnvelope;


def _make_wav(data, rate):
    """ Transform a numpy array to a PCM bytestring """
    import struct
    from io import BytesIO
    import wave

    try:
        import numpy as np

        data = np.array(data, dtype=float)
        if len(data.shape) == 1:
            nchan = 1
        elif len(data.shape) == 2:
            # In wave files,channels are interleaved. E.g.,
            # "L1R1L2R2..." for stereo. See
            # http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
            # for channel ordering
            nchan = data.shape[0]
            data = data.T.ravel()
        else:
            raise ValueError('Array audio input must be a 1D or 2D array')
        scaled = np.int16(data / np.max(np.abs(data)) * 32767).tolist()
    except ImportError:
        # check that it is a "1D" list
        idata = iter(data)  # fails if not an iterable
        try:
            iter(next(idata))
            raise TypeError('Only lists of mono audio are '
                            'supported if numpy is not installed')
        except TypeError:
            # this means it's not a nested list, which is what we want
            pass
        maxabsvalue = float(max([abs(x) for x in data]))
        scaled = [int(x / maxabsvalue * 32767) for x in data]
        nchan = 1

    fp = BytesIO()
    waveobj = wave.open(fp, mode='wb')
    waveobj.setnchannels(nchan)
    waveobj.setframerate(rate)
    waveobj.setsampwidth(2)
    waveobj.setcomptype('NONE', 'NONE')
    waveobj.writeframes(b''.join([struct.pack('<h', x) for x in scaled]))
    val = fp.getvalue()
    waveobj.close()

    return val