from .VisBeatImports import *

class Event(AObject):
    """Event (class): An event in time, either in video or audio
        Attributes:
            start: when the event starts
    """

    DIRECTION_FORWARD = 1;
    DIRECTION_BACKWARD = -1;
    DIRECTION_BOTH = 0;
    _EVENT_PHASE_BASE_RES = 8;

    def AOBJECT_TYPE(self):
        return 'Event';

    def __str__(self):
        return str(self.getAttributeDict());

    def __init__(self, start=None, type=None, weight=None, index=None, is_active=1, unrolled_start = None, direction=0, **kwargs):
        AObject.__init__(self, path=None);
        self.start = start;
        self.type=type;
        self.weight=weight;
        self.index = index;
        self.unrolled_start = unrolled_start;
        self.is_active = is_active;
        self.direction = direction;
        self.a_info.update(kwargs);

    def initializeBlank(self):
        AObject.initializeBlank(self);
        self.start = None;
        self.type = None;
        self.weight = None;
        self.index = None;
        self.unrolled_start = None;
        self.is_active = None;
        self.direction = None;


    def getAttributeDict(self):
        d = dict();
        d['start'] = self.start;
        d['type'] = self.type;
        d['weight'] = self.weight;
        d['index'] = self.index;
        d['unrolled_start'] = self.unrolled_start;
        d['is_active'] = self.is_active;
        d['direction'] = self.direction;
        return d;

    def toDictionary(self):
        d=AObject.toDictionary(self);
        d.update(self.getAttributeDict());
        return d;


    def initAttributesFromDictionary(self, d):
        self.start = d['start'];
        self.type = d.get('type');
        self.weight = d.get('weight');
        self.index = d['index'];
        self.unrolled_start = d.get('unrolled_start');
        self.is_active = d['is_active'];
        if (d.get('direction') is None):
            self.direction = 0;
        else:
            self.direction = d['direction'];

    def initFromDictionary(self, d):
        AObject.initFromDictionary(self, d);
        self.initAttributesFromDictionary(d);

    def clone(self, start=None):
        newe = Event();
        newe.initFromDictionary(self.toDictionary());
        if (start):
            newe.start = start;
        return newe;


    def _getIsSelected(self):
        return self.getInfo('is_selected');
    def _setIsSelected(self, is_selected):
        self.setInfo('is_selected', is_selected);

    def _getPhase(self, phase_resolution=None):
        if (self.getInfo('phase') is None):
            return -1;
        if (phase_resolution is None):
            return self.getInfo('phase');
        else:
            return self.getInfo('phase') * (phase_resolution / Event._EVENT_PHASE_BASE_RES);

    def _setPhase(self, phase, phase_resolution):
        if (phase_resolution is None):
            self.setInfo('phase', phase);
        else:
            self.setInfo('phase', phase * (Event._EVENT_PHASE_BASE_RES / phase_resolution));

    def _getBoundaryType(self):
        return self.getInfo('boundary_type');
    def _setBoundaryType(self, boundary_type):
        self.setInfo('boundary_type', boundary_type);

    @classmethod
    def _FromGUIDict(cls, gd, phase_resolution=None):
        e = cls();
        e.initAttributesFromDictionary(gd);
        e._setPhase(gd['phase'], phase_resolution=phase_resolution);
        e._setBoundaryType(gd.get('boundary_type'));
        e._setIsSelected(gd.get('is_selected'));
        return e;

    @classmethod
    def _FromGUIDicts(cls, gds, type=None):
        events = [];
        for gd in gds:
            ne = cls._FromGUIDict(gd);
            if (type is not None):
                ne.type = type;
            events.append(ne);
        return events;

    def _toGUIDict(self):
        '''
        note that is_active is switched to 0 or 1 for javascript/json
        :return:
        '''
        d = self.getAttributeDict();
        if(self.is_active):
            d['is_active'] = 1;
        else:
            d['is_active'] = 0;
        d['phase'] = self._getPhase();
        d['boundary_type'] = self._getBoundaryType();
        d['is_selected'] = self._getIsSelected();
        return d;

    @staticmethod
    def _ToGUIDicts(events, active=None):
        '''
        convert to dicts for javascript GUI
        :param events:
        :param active: if not none, all of the is_active flags will be set to this
        :return:
        '''
        startind = int(round(time.time() * 1000));
        starts = [];
        for e in range(len(events)):
            de = events[e]._toGUIDict();
            if (active is not None):
                assert (active is 0 or active is 1), "is_active must be 0 or 1";
                de['is_active'] = active;
            if (de.get('index') is None):
                de['index'] = startind + e;
            starts.append(de);
        return starts;


    def getUnrolledStartTime(self):
        if(self.unrolled_start is not None):
            return self.unrolled_start;
        else:
            return self.start;


    def getStartTime(self):
        return self.start;



    def getShifted(self, new_start_time):
        return Event(self.start-new_start_time, type=self.type, weight=self.weight, index=self.index);




    @staticmethod
    def GetUnrolledList(event_list, assert_on_folds=None):
        out_list = [];
        event0 = event_list[0].clone();
        event0.unrolled_start = event0.start;
        event0.index = 0;
        out_list.append(event0);
        for e in range(1,len(event_list)):
            newe = event_list[e].clone();
            if(assert_on_folds):
                assert(newe.start>=out_list[-1].start), 'FOLD (non-monotonic event list) DETECTED WHEN NOT ALLOWED!!!\n see Event.GetUnrolledList'
            newe.unrolled_start = out_list[-1].unrolled_start+np.fabs(newe.start-out_list[-1].start);
            newe.index = e;
            out_list.append(newe);
        return out_list;

    @staticmethod
    def NewFromIndices(event_list, inds):
        out_list = [];
        for ei in inds:
            out_list.append(event_list[ei].clone());
        return out_list;

    @staticmethod
    def RollToNOld(events, n_out, momentum = 0.25):
        inds = [0];
        step = 1;
        n_events = len(events);
        for b in range(1,n_out):
            lastind = inds[-1];
            if ((n_out-b)<(n_events-lastind) or lastind==0):
                inds.append(lastind+1);
                step = 1;
            elif(lastind==(len(events)-1)):
                inds.append(len(events)-2);
                step = -1;
            else:
                foreward_p = 0.5+momentum;
                roll = np.random.rand();
                if(roll<foreward_p):
                    inds.append(lastind+step);
                else:
                    inds.append(lastind-step);
                    step = -step;
        print(inds);
        return Event.GetUnrolledList(Event.NewFromIndices(events, inds));

    @staticmethod
    def UnfoldToN(events, n_out, momentum=0.25):
        # This weird way of implementing is the result of stripping non-visbeat stuff out of the original code...
        return Event.RollToN(events=events, n_out=n_out, momentum=momentum);



    @staticmethod
    def GetDirectedLinks(events):
        links = [{}]
        links[0]['prev_f']=None;
        links[0]['prev_b'] = None;
        lastf = None;
        lastb = None;
        if(events[0].direction<1):
            lastb = 0;
        if(events[0].direction>-1):
            lastf = 0;

        for ei in range(1,len(events)):
            links.append({});
            links[ei]['prev_f']=lastf;
            links[ei]['prev_b']=lastb;

            lastbu = lastb;
            lastfu = lastf;
            if(lastbu is None):
                lastbu = 0;
            if(lastfu is None):
                lastfu = 0;


            if(events[ei].direction<1):
                for lb in range(lastbu,ei):
                    links[lb]['next_b']=ei;
                    # print("ei is {} and lastbu is {}, lastb is {}".format(ei, lastbu, lastb));
                lastb = ei;
            if (events[ei].direction>-1):
                for lf in range(lastfu, ei):
                    links[lf]['next_f'] = ei;
                    # print("ei is {} and lastfu is {}, lastf is {}".format(ei, lastfu, lastf));
                lastf = ei;

        return links;


    @staticmethod
    def RollToN(events, n_out, start_index = 0, momentum=0.1):
        links = Event.GetDirectedLinks(events);
        inds = [start_index];
        step = 1;
        n_events = len(events);
        for b in range(1, n_out):
            lastind = inds[-1];
            foreward_p = 0.5 + momentum;
            roll = np.random.rand();

            if (roll < foreward_p):
                step = step;
            else:
                step = -step;

            if(links[lastind].get('prev_b') is None):
                step = 1;
            if(links[lastind].get('next_f') is None):
                step = -1;

            if(step>0):
                inds.append(links[lastind]['next_f']);
            else:
                inds.append(links[lastind]['prev_b']);
        # print(inds);
        return Event.GetUnrolledList(Event.NewFromIndices(events, inds));



    @staticmethod
    def Clone(event_list):
        out_list = [];
        for ei in event_list:
            out_list.append(ei.clone());
        return out_list;

    @staticmethod
    def SetDirections(event_list, direction):
        for e in range(len(event_list)):
            event_list[e].direction = direction;
        return event_list;



    @classmethod
    def FromStartTimes(cls, starts, type=None):
        events = [];
        for s in starts:
            events.append(cls(start=s, type=type));
        return events;

    @classmethod
    def FromStartsAndWeights(cls, starts, weights, type=None):
        events = [];
        assert(len(starts)==len(weights)), 'Event.FromStartsAndWeights got {} starts and {} weights'.format(len(starts), len(weights));
        for s in range(len(starts)):
            events.append(cls(start=starts[s], weight=weights[s],type=type));
        return events;

    @staticmethod
    def ToStartTimes(events):
        starts = np.zeros(len(events));
        for e in range(len(events)):
            starts[e]=events[e].start;
        return starts;

    @staticmethod
    def ToWeights(events):
        weights = np.zeros(len(events));
        for e in range(len(events)):
            weights[e] = events[e].weight;
        return weights;

    #endpoint is false if events are already placed at beginning and end
    @staticmethod
    def RepeatToLength(events, n, endpoints=False):
        if(n<len(events)):
            return events[:n];
        if(n==len(events)):
            return events;

        if(endpoints):
            print("HAVE NOT IMPLEMENTED ENDPOINTS VERSION OF REPEAT")

        dup_index = 1;
        while(len(events)<n):
            interval = events[dup_index].start-events[dup_index-1].start;
            last_time = events[-1].start;
            newevent = events[dup_index].clone(start=last_time+interval);
            events.append(newevent);
            dup_index=dup_index+1;
        return events;

    @staticmethod
    def Double(events):
        doubled = [];
        for e in range(1,len(events)):
            halfstart = 0.5*(events[e].start+events[e-1].start);
            doubled.append(events[e].clone(start=halfstart));
            doubled.append(events[e]);
        return doubled;

    @staticmethod
    def Half(events, offset=0):
        if(offset is None):
            offset=0;

        halved = [];
        for e in range(1,len(events)):
            if((e%2)==offset):
                halved.append(events[e]);
        return halved;

    @staticmethod
    def SubdivideIntervals(events, extra_samples_per_interval=1):
        newe = [];
        samples = np.linspace(0,1,extra_samples_per_interval+2)[1:-1];
        newe.append(events[0]);
        for e in range(1, len(events)):
            for s in samples:
                newstart = events[e].start*s+events[e-1].start*(1.0-s);
                newe.append(events[e].clone(start=newstart));
            newe.append(events[e].clone())
        return newe;

    @staticmethod
    def Third(events, offset=0):
        if (offset is None):
            offset = 0;
        third = [];
        for e in range(len(events)):
            if (not ((e+offset) % 3)):
                third.append(events[e]);
        return third;

    @staticmethod
    def SubsampleEveryN(events, n, offset=0):
        if (offset is None):
            offset = 0;
        newe = [];
        for e in range(len(events)):
            if (not ((e + offset) % n)):
                newe.append(events[e]);
        return newe;

    # @staticmethod
    # def FromSignalBeats(signal, sampling_rate, event_type=None, **kwargs):
    #     # seg = [0,100]
    #     # xvals = range(seg[1]-seg[0])
    #     xvals = np.arange(len(signal));
    #     xvals = xvals*truediv(1.0, sampling_rate);
    #
    #     vis_tempo, beatinds = librosa.beat.beat_track(y=None, sr=sampling_rate, onset_envelope=signal, hop_length=1, start_bpm=100.0, tightness=80,
    #                                                   trim=True, bpm=None, units='frames');
    #     peaktimes = xvals[beatinds.astype(int)];
    #     peakvals = signal[beatinds.astype(int)];
    #     return Event.FromStartsAndWeights(starts=peaktimes, weights=peakvals, type=event_type);


    @staticmethod
    def FromSignalPeaks(signal, sampling_rate, event_type=None, index_offset=None, **kwargs):
        xvals = np.arange(len(signal));
        xvals = xvals*truediv(1.0, sampling_rate);

        time_params = dict(
            pre_max_time=0.2,
            post_max_time=0.2,
            pre_avg_time=0.2,
            post_avg_time=0.2,
            wait_time=0.1,
        )
        tp_keys = list(time_params.keys());
        time_params.update(kwargs);

        delta = kwargs.get('delta');
        if (delta is None):
            delta = 0.2;

        print(delta)

        for p in tp_keys:
            time_params[p] = int(round(sampling_rate * time_params[p]));

        dparams = dict(
            pre_max=time_params['pre_max_time'],
            post_max=time_params['post_max_time'],
            pre_avg=time_params['pre_avg_time'],
            post_avg=time_params['post_avg_time'],
            wait=time_params['wait_time'],
            delta=delta
        )

        # print(dparams)
        peakinds = librosa.util.peak_pick(x=signal, **dparams);
        # print(peakinds)
        if(not len(peakinds)):
            return [];
        if(index_offset is not None):
            print(('index offset {}'.format(index_offset)))
            for pi in range(len(peakinds)):
                newi = peakinds[pi]+index_offset;
                if(newi>=0 and newi<len(xvals)):
                    print(('{} to {}'.format(peakinds[pi], newi)))
                    peakinds[pi]=newi;


        peaktimes = xvals[peakinds];
        peakvals = signal[peakinds];
        return Event.FromStartsAndWeights(starts=peaktimes, weights=peakvals, type=event_type);


    @staticmethod
    def PlotEventMatches(source_events, target_events, source_in=None, target_in=None):
        t_height = 0.75;
        s_height = 0.25;
        if ((source_in is not None) and (target_in is not None)):
            n_in = min(len(source_in), len(target_in));
            evt = Event.ToStartTimes(target_in[:n_in]);
            evs = Event.ToStartTimes(source_in[:n_in]);
            plt.plot(evt, np.ones(len(evt)) * t_height, 'o');
            plt.plot(evs, np.ones(len(evs)) * s_height, 'o');

        s_et = Event.ToStartTimes(source_events);
        t_et = Event.ToStartTimes(target_events);

        for ev in range(len(target_events)):
            plt.plot([s_et[ev], t_et[ev]], [s_height, t_height], '-');

    @staticmethod
    def GetWithFirstEventAt(events, first_event_time):
        oute = [];
        shift_time = events[0].start-first_event_time;
        for e in events:
            oute.append(e.getShifted(new_start_time=shift_time));
        return oute;

    @staticmethod
    def GetWithStartTimesShifted(events, new_start_time):
        oute = [];
        for e in events:
            oute.append(e.getShifted(new_start_time=new_start_time));
        return oute;

    @staticmethod
    def GetScaled(events, scale):
        oute = [];
        for e in events:
            oute.append(Event(e.start*scale, weight=e.weight, type=e.type, index=e.index));
        return oute;

    @staticmethod
    def GetScaledAndStartingAt(events, scale, starting_at):
        ev = Event.GetScaled(events, scale);
        ev = Event.GetWithFirstEventAt(ev, starting_at);
        return ev;

    @staticmethod
    def ClosestToTargetMatch(source_events, target_events):
        def _closest_e(target_e):
            def sortfunc_e(source_e):
                return math.fabs(source_e[1].start-target_e.start);
            return sortfunc_e;
        def _getinds(s_sort):
            inds = [];
            for s in s_sort:
                inds.append(s[0]);
            return inds;
        def _getevents(s_sorted):
            evnts = [];
            for s in s_sort:
                evnts.append(s[1]);
            return evnts;

        S = [];
        T = [];
        for s in range(len(source_events)):
            S.append([s, source_events[s]]);
        for t in range(len(target_events)):
            T.append([t, target_events[t]]);

        S_sorted = [];
        for t in T:
            S_sorted.append(sorted(S, key=_closest_e(t[1])));

        s_out_inds = [-1]*len(T);
        match_map = {};
        for ti in range(len(S_sorted)):
            # print(S_sorted[ti][0][0]);
            matched_source_event_i = S_sorted[ti][0][0];#sorted lists, first element, index value for index
            existing_match = match_map.get(matched_source_event_i);
            matchdist = math.fabs(source_events[matched_source_event_i].start - target_events[ti].start);
            if(existing_match is not None):
                if(matchdist<existing_match[1]):
                    # print('overrule')
                    s_out_inds[existing_match[0]]=-1;
                    s_out_inds[ti]=matched_source_event_i;
                    # print("ti {}\nmatched {}\n".format(ti, matched_source_event_i));
                    match_map[matched_source_event_i]=[ti, matchdist];
            else:
                match_map[matched_source_event_i] = [ti, matchdist];
                s_out_inds[ti]=matched_source_event_i;

        S_out = [];
        T_out = [];
        # print("s_out_inds: {}".format(s_out_inds))
        for e in range(len(s_out_inds)):
            if(s_out_inds[e]>=0):
                T_out.append(target_events[e]);
                S_out.append(source_events[s_out_inds[e]]);

        return S_out, T_out;

    @staticmethod
    def Sort(event_list, func=None):
        assert(func is None or func=='start' or func=='time'), "have not implemented sort by {}".format(func);
        event_list.sort(key=lambda x: x.start);
        return event_list;

    @staticmethod
    def GetSorted(event_list, func=None):
        clone = Event.Clone(event_list);
        Event.Sort(clone, func=func);
        return clone;

    @staticmethod
    def GetWithTwoWayMerged(event_list, merge_window = 0.1):
        event_list_sorted = Event.GetSorted(event_list);
        new_events = [];
        new_events.append(event_list_sorted[0].clone());
        for ei in range(1,len(event_list_sorted)):
            thise = event_list_sorted[ei];
            if((thise.start-new_events[-1].start)<merge_window):
                if((thise.direction+new_events[-1].direction)==0):
                    new_events[-1].start = 0.5*(thise.start+new_events[-1].start);
                    new_events[-1].direction = 0;
            else:
                new_events.append(thise.clone());
        return new_events


    @staticmethod
    def ApplyIndices(event_list):
        for e in range(len(event_list)):
            event_list[e].index = e;

    @staticmethod
    def PlotSignalAndEvents(signal, sampling_rate, events, time_range = None, ylim=None, **kwargs):
        times = np.arange(len(signal));
        times = times*truediv(1.0,sampling_rate);

        if (kwargs.get('other_events')):
            oet, oev = eventsToTimes(kwargs.get('other_events'), time_range);
            plt.plot(oet, oev, 'x');

        if(events is not None):
            btimes = Event.ToStartTimes(events);
            binds = np.round(btimes*sampling_rate).astype(int);
            bvals = signal[binds];

            mplt = plt.plot(times, signal, '-', btimes, bvals, 'o');
        else:
            mplt = plt.plot(times, signal, '-');

        if(time_range is not None):
            plt.xlim(time_range[0], time_range[1])
        if(ylim is not None):
            plt.ylim(ylim);
        # plt.show()
        return mplt;
