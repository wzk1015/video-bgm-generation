from .Event import *

class EventList(AObject):
    """Event (class): An event in time, either in video or audio
        Attributes:
            start: when the event starts
    """

    def AOBJECT_TYPE(self):
        return 'EventList';

    def __init__(self, events=None):
        AObject.__init__(self, path=None);
        if(events is not None):
            self.events = events;


    def initializeBlank(self):
        AObject.initializeBlank(self);
        self.events = [];

    def list(self):
        return self.events;

    def Clone(self):
        return EventList(Event.Clone(self.events));

    def toDictionary(self):
        d=AObject.toDictionary(self);
        d['events']=self.serializeEvents();
        return d;

    def serializeEvents(self):
        re = [];
        for e in self.events:
            re.append(e.toDictionary());
        return re;

    def getActiveEvents(self):
        active_events = [];
        for e in self.events:
            if(e.is_active):
                active_events.append(e);
        return active_events;

    def initFromDictionary(self, d):
        AObject.initFromDictionary(self, d);
        events = d['events'];
        for e in events:
            newe = Event();
            newe.initFromDictionary(e);
            self.events.append(newe);

    def unroll(self, assert_on_folds=None):
        newlist = Event.GetUnrolledList(event_list=self.events, assert_on_folds=assert_on_folds);
        self.events = newlist;

    def getUnrolled(self, assert_on_folds=None):
        return EventList(Event.GetUnrolledList(event_list=self.events, assert_on_folds=assert_on_folds));

    def getFromIndices(self, inds):
        return EventList(Event.NewFromIndices(event_list=self.events, inds=inds));

    def getRolledToN(self, n_out, momentum = 0.1):
        return EventList(Event.RollToN(self.events, n_out=n_out, momentum=momentum));


    def toStartTimes(self):
        return Event.ToStartTimes(self.events);

    def _toGUIDicts(self):
        return Event._ToGUIDicts(self.events);

    @staticmethod
    def _FromGUIDicts(gds, type=None):
        events = Event._FromGUIDicts(gds, type=type);
        return EventList(events);


    @staticmethod
    def FromJSON(json_path=None):
        if (json_path is None):
            json_path = fileui.GetFilePath();
        elist = EventList();
        elist.loadFromJSON(json_path=json_path);
        return elist;

    @staticmethod
    def FromStartTimes(starts, type=None, event_class=None):
        elist = EventList();
        if(event_class is None):
            event_class = Event;
        for s in starts:
            elist.events.append(event_class(start=s, type=type));
        return elist;


    def toWeights(self):
        return Event.ToWeights(self.events);

    def getDoubled(self):
        return EventList(Event.Double(self.Clone().events));

    def getHalved(self, offset=0):
        return EventList(Event.Half(self.Clone().events, offset=offset));

    def getThirded(self, offset=0):
        return EventList(Event.Third(self.Clone().events, offset=offset));

    @staticmethod
    def FromSignalPeaks(**kwargs):
        return EventList(Event.FromSignalPeaks(**kwargs));

    @staticmethod
    def PlotEventMatches(source_elist, target_elist, source_in=None, target_in=None):
        return Event.PlotEventMatches(source_events=source_elist.events, target_events=target_elist.events, source_in=source_in, target_in=target_in);

    def getWithFirstEventAt(self, first_event_time):
        return EventList(Event.GetWithFirstEventAt(events=self.events, first_event_time=first_event_time));

    def getWithStartTimesShifted(self, new_start_time):
        return EventList(Event.GetWithStartTimesShifted(events = self.events, new_start_time=new_start_time));