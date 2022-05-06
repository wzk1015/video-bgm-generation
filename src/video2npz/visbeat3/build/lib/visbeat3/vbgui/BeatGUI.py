from visbeat3.AImports import *

VIEWER_INSTALLED = 1;
try:
    import vbwidget as Viewer
except ImportError as e:
    VIEWER_INSTALLED = 0;
    AWARN("VBViewer not installed. Consider installing for full functionality.")

from ..TimeSignal import *
from ..EventList import *



#this is what media should call to get its gui object
def media_GUI_func(self):
    if (self._gui is None):
        self._gui = BeatGUI();
        self._gui.media = self;
    return self._gui;

class BeatGUI(AObject):
    """

    """
    def AOBJECT_TYPE(self):
        return 'BeatGUI';

    def __init__(self, media=None, path=None, clear_temp=None):
        """If you provide a directory, it will look for a existing AFileManager.json in that directory, or create one if it does not already exist.
            If you provide a json, it will use that json, unless the json doesn't exist, in which case it will complain...
        """
        AObject.__init__(self, path=path);
        if(media is not None):
            self.media = media;

    def initializeBlank(self):
        AObject.initializeBlank(self);
        self._widget = None;
        self._media = None;


    def getJSONName(self):
        return self.AOBJECT_TYPE()+".json";


    #########

    # <editor-fold desc="Property: 'media'">
    @property
    def media(self):
        return self._getMedia();
    def _getMedia(self):
        return self._media;
    @media.setter
    def media(self, value):
        self._setMedia(value);
    def _setMedia(self, value):
        self._media = value;
    # </editor-fold>

    # <editor-fold desc="Property: 'media_type'">
    @property
    def media_type(self):
        return self._getMediaType();
    def _getMediaType(self):
        if(self.media is None):
            return None;
        else:
            return self.media.AObjectType();
    # </editor-fold>

    # <editor-fold desc="Property: 'widget'">
    @property
    def widget(self):
        return self._getWidget();
    def _getWidget(self):
        if (self._widget is None):
            self._widget = Viewer.VBVSignal();
        return self._widget;
    @widget.setter
    def widget(self, value):
        self._setWidget(value);
    def _setWidget(self, value):
        self._widget = value;
    # </editor-fold>

    # <editor-fold desc="Property: 'frame_rate'">
    @property
    def frame_rate(self):
        return self._getFrameRate();
    def _getFrameRate(self):
        gfr = self.widget.frame_rate;
        if (gfr is None):
            media = self.media;
            if (media is not None):
                gfr = media.getFrameRate();
        return gfr;
    @frame_rate.setter
    def frame_rate(self, value):
        self._setFrameRate(value);
    def _setFrameRate(self, value):
        self.widget.frame_rate = float(value);
    # </editor-fold>


    # <editor-fold desc="Property: 'frame_offset'">
    @property
    def frame_offset(self):
        return self._getFrameOffset();
    def _getFrameOffset(self):
        return self.widget.frame_offset;
    @frame_offset.setter
    def frame_offset(self, value):
        self._setFrameOffset(value);
    def _setFrameOffset(self, value):
        self.widget.frame_offset = value;
    # </editor-fold>

    def run(self, local_saliency=None, frame_rate = None, eventlist = 'default', frame_offset=None):
        if(frame_rate is None):
            # self.widget.frame_rate = float(self.getMedia().getFrameRate());
            self.frame_rate = self.media._getFrameRate();
        else:
            # self.widget.frame_rate = float(frame_rate);
            self.frame_rate = frame_rate;

        if(local_saliency is None):
            self.widget.signal = self.media.getLocalRhythmicSaliency().tolist();
            # self.widget.signal = self.getBothWayVisualImpactEnvelope(highpass_window_seconds=None, force_recompute = True).tolist();
        else:
            self.widget.signal = local_saliency.tolist();

        if(frame_offset is None):
            self.frame_offset = 0;
        elif(frame_offset is 'guess'):
            self.frame_offset = self.guessFrameOffset();
        else:
            self.frame_offset = frame_offset;

        if(eventlist is None):
            self.widget.events = [];
        elif(eventlist == 'default'):
            self.widget.events = EventList._toGUIDicts(self.media.getEventList());
        else:
            self.widget.events = EventList._toGUIDicts(eventlist);
        self.widget.data_string = self.media.getStringForHTMLStreamingBase64();
        return self.widget;


    def guessFrameOffset(self):
        if(isinstance(self.media, Video)):
            return self.media.reader.get_length() - self.media.n_frames();
        else:
            return 0;


    def deactivateAllEvents(self):
        newes = []
        gevents = self.getEventDicts();
        for e in gevents:
            newe = e;
            newe['is_active']=0;
            newes.append(newe);
        self.widget.events = [];
        self.widget.events = newes;

    def activateAllEvents(self):
        newes = []
        gevents = self.getEventDicts();
        for e in gevents:
            newe = e;
            newe['is_active'] = 1;
            newes.append(newe);
        self.widget.events = [];
        self.widget.events = newes;

    def activatePattern(self, pattern=None, prefix=None, apply_to_active=None):
        assert(pattern), "must provide pattern to activate"
        newes = []
        gevents = self.getGUIEventDicts();
        counter = 0;
        prefix_length = 0;
        if(prefix_length is not None):
            prefix_length = len(prefix);
        for i, e in enumerate(gevents):
            if (apply_to_active):
                if (e.get('is_active')):
                    if (counter < prefix_length):
                        e['is_active']=prefix[counter];
                    else:
                        e['is_active'] = pattern[(counter - prefix_length) % len(pattern)];
                    counter = counter + 1;
                else:
                    print(("Skipping beat {}, inactive".format(i)));
            else:
                if (i < prefix_length):
                    e['is_active'] = prefix[i];
                else:
                    e['is_active'] = pattern[(i - prefix_length) % len(pattern)];
            newes.append(e);

        self.widget.events = [];
        self.widget.events = newes;


    def shiftEventsByNFrames(self, n_frames=None):
        assert(n_frames), "must provide number of frames to shift by"
        newes = []
        gevents = self.getEventDicts();
        sample_step = np.true_divide(1.0,self.getFrameRate());
        for e in gevents:
            newe = e;
            newe['start'] = newe['start']+n_frames*sample_step;
            newes.append(newe);
        self.widget.events = [];
        self.widget.events = newes;


    def getActiveEventTimes(self):
        gevents = self.getEventDicts(active_only=True);
        revents = []
        for e in gevents:
            revents.append(e.get('time'));
        return np.asarray(revents);

    def getEventTimes(self):
        gevents = self.getEventDicts();
        revents = []
        for e in gevents:
            revents.append(e.t);
        return np.asarray(revents);

    def getEvents(self, active_only=None):
        return Event._FromGUIDicts(self.getEventDicts(active_only = active_only));

    def getEventList(self, active_only=None):
        elist = EventList._FromGUIDicts(self.getEventDicts(active_only=active_only));
        elist.setInfo(label='html_frame_offset', value=self.getFrameOffset());
        return elist;

    def getActiveEvents(self):
        return self.getEvents(active_only=True);

    def getEventDicts(self, active_only = None):
        gevents = self.widget.events[:];
        if(not active_only):
            return gevents;
        else:
            nevents = []
            for e in gevents:
                if(e.get('is_active')):
                    nevents.append(e);
            return nevents;

    def saveEvents(self, save_path = None):
        elist = self.getEventList(active_only=False);
        if(save_path is not None):
            elist.writeToJSON(json_path=save_path);
            self.widget.last_save_path = save_path;
        else:
            save_path = self.widget.last_save_path;
            if(save_path is None):
                save_path = uiGetSaveFilePath(file_extension='.json');
            if(save_path is not None):
                elist.writeToJSON(json_path=save_path);
                self.widget.last_save_path = save_path;

    def saveEventsAs(self, save_path = None):
        elist = self.getEventList(active_only=False);
        if (save_path is not None):
            elist.writeToJSON(json_path=save_path);
            self.widget.last_save_path = save_path;
            print(('savepath not none {}'.format(save_path)))
        else:
            save_path = uiGetSaveFilePath(file_extension='.json');
            print(('savepath from ui {}'.format(save_path)))
            if (save_path is not None):
                print(('save path from ui {}'.format(save_path)));
                elist.writeToJSON(json_path=save_path);
                self.widget.last_save_path = save_path;
        print(save_path)

    def setEvents(self, events):
        self.widget.events = Event._ToGUIDicts(events);

    def setEventList(self, event_list):
        if(event_list.getInfo('html_frame_offset') is not None):
            self.widget.frame_offset = event_list.getInfo('html_frame_offset');
        self.widget.events = event_list._toGUIDicts();


    def loadEvents(self, load_path = None):
        if(load_path is None):
            load_path = uiGetFilePath();
        elist = EventList();
        elist.loadFromJSON(json_path=load_path);
        self.setEventList(event_list = elist);



    def getEventListWithSelectedSegments(self):
        eventlist = self.getEventList();
        events = eventlist.events;
        segments = [];
        for i, e in enumerate(events):
            if(e.direction>-1): # meaning not a back beat
                newseg = [];
                for si in range(i, len(events)):
                    newseg.append(si);
                    if(events[si].direction<0): # meaning a back beat
                        break;
                segments.append(newseg);
        eventlist.setInfo(label='selected_segments', value=segments);
        return eventlist;





