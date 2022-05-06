
from .Video import *
from .AudioClip import *

class VideoClip(Video):
    """VideoClip (class): A segment of a video, and a bunch of convenience functions to go with it.
        Attributes:
            start: The name of the video
            end: The framerate of the video
    """

    def AOBJECT_TYPE(self):
        return 'VideoClip';

    def __init__(self, video=None, start=None, end=None, clip_to_frame=True, path=None):
        """If a video is provided, we don't want to reload the video from disk. If a path is provided, we have no choice.
        """
        assert(not (video and path)), "provided both video object and path to VideoClip constructor."
        assert((start is not None) and (end is not None)), "must provide start time and end time to AudioClip constructor"
        Video.__init__(self, path = path);
        self.initializeBlank();
        self.start = start;
        self.end = end;

        if(video):
            self.setPath(video.getPath());
            self.reader = video.reader;
            #self.writer = None; #this only comes up if we write later
            self.sampling_rate=video.sampling_rate;
            self.num_frames_total=video.num_frames_total;
            time_per_frame=truediv(1.0,self.sampling_rate);
            if(clip_to_frame):
                self.start=time_per_frame*math.floor(truediv(self.start,time_per_frame));
                self.end=time_per_frame*math.floor(truediv(self.end,time_per_frame));
            if(video.name):
                self.name = video.name+"_{}_{}".format(start,end);
            self.audio = video.audio.AudioClip(start=start, end=end);
        else:
            assert(False),"must provide video to VideoClip init"

    def initializeBlank(self):
        Video.initializeBlank(self);
        self.start = None;
        self.end = None;


    # def readFrameBasic(self, i):
    #     return Video.getFrame(self, float(i)+self.start);

    def getFrameLinearInterp(self, f):
        return Video.getFrameLinearInterp(self, float(f)+self.start*self.sampling_rate);

    def getFrame(self, f):
        return Video.getFrame(self, float(f)+self.start*self.sampling_rate);

    def getDuration(self, round_to_frames=False):
        if(not round_to_frames):
            return self.end-self.start;
        else:
            return truediv(self.n_frames(), self.sampling_rate);

    def n_frames(self):
        #return self.getLastFrameIndex()-self.getFirstFrameIndex();
        return math.ceil((self.end-self.start)*self.sampling_rate);

    # def getFirstFrameIndex(self):
    #     return math.floor(self.start*self.sampling_rate);
    #
    # def getLastFrameIndex(self):
    #     #I think floor is still right here, because the times mark beginnings of frames
    #     return math.floor(self.end*self.sampling_rate);

    def getStartTime(self):
        return self.start;

    def getEndTime(self):
        return self.end;

    def getMPYClip(self, get_audio=True):
        return mpy.VideoFileClip(self.getPath(), audio=get_audio).subclip(self.getStartTime(), self.getEndTime());

    def play(self):
        if(ISNOTEBOOK):
            print("Playing video:")
            IPython.display.display(self.getMPYClip().ipython_display(fps=self.sampling_rate, maxduration=self.getDuration()+1));
        else:
             print("HOW TO PLAY VIDEO? NOT A NOTEBOOK.")
