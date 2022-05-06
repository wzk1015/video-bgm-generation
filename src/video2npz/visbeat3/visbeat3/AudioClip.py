from .Audio import *

class AudioClip(Audio):
    """AudioClip (class): A segment of a video, and a bunch of convenience functions to go with it.
        Attributes:
            start: The name of the video
            end: The framerate of the video
    """

    def VBOBJECT_TYPE(self):
        return 'AudioClip';

    def __init__(self, audio=None, start=None, end=None, path=None):
        assert(not (audio and path)), "provided both video object and path to VideoClip constructor."
        # assert((start is not None) and (end is not None)), "must provide start time and end time to AudioClip constructor"
        Audio.__init__(self, path=path);
        #self.initializeBlank();#gets called by parent
        if(start is None):
            start = 0;
        if(end is None):
            if(audio is not None):
                end = audio.getDuration();
            else:
                end = truediv(len(self.x), self.sampling_rate);

        self.start = start;
        self.end = end;


        if(audio):
            self.setPath(audio.getPath());
            self.x = audio.x;
            self.sampling_rate=audio.sampling_rate;
            self.n_channels=audio.n_channels;
            stereo = audio.getStereo();
            if(stereo):
                self.a_info['stereo_signal']=stereo;
        if(self.sampling_rate):
            self._pull_clip_potion();


    def _pull_clip_potion(self):
        startsample = math.floor(self.sampling_rate * self.start);
        endsample = math.ceil(self.sampling_rate * self.end);
        startinsample = max(startsample, 0);
        endinsample = min(endsample, len(self.x));
        self.clipped = self.x[int(startinsample):int(endinsample)];
        if (startsample < 0):
            self.clipped = np.concatenate((np.zeros(int(-startsample)), self.clipped));
        if (endsample > len(self.x)):
            self.clipped = np.concatenate((np.zeros(int(endsample - len(self.x))), self.clipped));


    def resample(self, sampling_rate):
        Audio.resample(self, sampling_rate);
        self._pull_clip_potion();

    def initializeBlank(self):
        Audio.initializeBlank(self);
        self.start = None;
        self.end = None;
        self.resampled = None;
        self.clipped = None;

    def getSignal(self, resample=False):
        if(self.resampled):
            return self.resampled;
        if(resample):
            assert(False), "haven't implemented audio clip resampling yet.";
        return self.clipped
