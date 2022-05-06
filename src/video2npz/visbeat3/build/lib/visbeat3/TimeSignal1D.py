from .TimeSignal import *
import math

class TimeSignal1D(TimeSignal):
    """TimeSignal1D (class): A time signal, and a bunch of convenience functions to go with it.
        Attributes:
            sampling_rate: the sampling rate
            x:  the time signal
    """

    def VBOJECT_TYPE(self):
        return 'TimeSignal1D';

    def __init__(self, path=None, sampling_rate = None, x=None):
        TimeSignal.__init__(self, path=path, sampling_rate=sampling_rate);
        if(x is not None):
            self.x = x;
        #self.initializeBlank(); # will be called by parent

    def initializeBlank(self):
        TimeSignal.initializeBlank(self);#YES KEEP call through weird loops
        self.x = None;

    def getSignal(self, resampled=False):
        return self.x;

    def getSignalSegment(self, time_range):
        signal = self.getSignal();
        seg_start = int(time_range[0]*self.sampling_rate);
        seg_end = int(time_range[1]*self.sampling_rate);
        return signal[seg_start:seg_end];

    def getFullSignal(self):
        return self.x;

    def getSampleAtTime(self, f):
        prev_sample = self.x[int(math.floor(f))];
        next_sample = self.x[int(math.ceil(f))];
        sample_progress = f-np.floor(f);
        return (next_sample*sample_progress)+(prev_sample*(1.0-sample_progress));

    def getSampleAtIndex(self, i):
        return self.x[i];

    def getDuration(self):
        return truediv(len(self.getSignal()), self.sampling_rate);

    def setValueRange(self, value_range=None):
        if(value_range is None):
            value_range = [0,1];
        data = self.x[:];
        currentscale = np.max(data)-np.min(data);
        data = (data/currentscale)*(value_range[1]-value_range[0]);
        data = data-np.min(data)+value_range[0]
        self.x = data;

    def setMaxAbsValue(self, max_abs_val=1.0):
        data = self.x[:];
        currentscale = np.max(np.fabs(data));
        data = (data/currentscale)*max_abs_val;
        self.x = data;
