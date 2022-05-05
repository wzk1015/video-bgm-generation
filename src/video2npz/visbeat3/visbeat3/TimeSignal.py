from .VBObject import *
from .Event import *
import math
from operator import truediv
from .VisBeatImports import *

class TimeSignal(VBObject):
    """TimeSignal (class): A time signal, and a bunch of convenience functions to go with it.
        Attributes:
            sampling_rate: the sampling rate
            visualizations dictionary of visualizations. (removed - too many dependencies)
    """

    def VBBJECT_TYPE(self):
        return 'TimeSignal';

    def __init__(self, path=None, sampling_rate = None):
        VBObject.__init__(self, path=path);
        # self.initializeBlank();
        # self.visualizations = AFuncDict(owner=self, name='visualizations');
        # self.visualizations.functions.update(self.VIS_FUNCS);
        if(sampling_rate):
            self.sampling_rate=sampling_rate;

    def initializeBlank(self):
        VBObject.initializeBlank(self);
        self.sampling_rate = None;


    # <editor-fold desc="Property: 'frame_rate'">
    @property
    def frame_rate(self):
        return self._getFrameRate();
    def _getFrameRate(self):
        raise NotImplementedError;
    # </editor-fold>


    def getSampleAtTime(self, f):
        prev_sample = self.getSampleAtIndex(math.floor(f));
        next_sample = self.getSampleAtIndex(math.ceil(f));
        sample_progress = f-np.floor(f);
        return (next_sample*sample_progress)+(prev_sample*(1.0-sample_progress));

    def getSampleAtIndex(self, i):
        return self.getTimeForIndex();

    def getDuration(self):
        assert(False), "getDuration must be implemented for subclass of TimeSignal"

    def getSampleDuration(self):
        return 1.0/self.sampling_rate;

    def getTimeForIndex(self, i):
        return i*self.getSampleDuration();

    # def toDictionary(self):
    #     d = VBObject.toDictionary(self);
    #     assert(False), "haven't implemented toDictionary for {} yet".format(self.VBOBJECT_TYPE())
    #     #serialize class specific members
    #     return d;

    # def initFromDictionary(self, d):
    #     VBObject.initFromDictionary(self, d);
    #     assert(False), "haven't implemented initFromDictionary for {} yet".format(self.VBOBJECT_TYPE())
    #     #do class specific inits with d;
