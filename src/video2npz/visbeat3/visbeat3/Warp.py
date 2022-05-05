

from .VisBeatImports import *
from .EventList import *
import math

DEFAULT_LEAD_TIME = 0;
DEFAULT_TAIL_TIME = 0;

class Warp(AObject):
    """Warp (class): defines how one time signal should be warped to another. Given primarily as source/target events to be matched.
        Attributes:
            source_events: source events
            target_events: target events
    """

    def VBOBJECT_TYPE(self):
        return 'Warp';

    def __init__(self, path=None):
        AObject.__init__(self, path=path);
        if(path):
            self.loadFile();


    @staticmethod
    def FromEvents(source_events, target_events):
        w = Warp();
        sevents = source_events;
        if(isinstance(source_events, EventList)):
            sevents = source_events.events;

        tevents = target_events;
        if(isinstance(target_events, EventList)):
            tevents = target_events.events;

        w.source_events=sevents;
        w.target_events=tevents;
        # w.repeatShorterEvents();
        return w

    @staticmethod
    def FromEventLists(source_eventlist, target_eventlist):
        w = Warp();
        w.source_events = source_eventlist.events;
        w.target_events = target_eventlist.events;
        # w.repeatShorterEvents();
        return w


    def initializeBlank(self):
        AObject.initializeBlank(self);
        self.source_events = [];
        self.target_events = [];
        # self.a_info['WarpType'] = 'Linear';
        self.warp_func = None;  # Warp.LinearInterp;
        # self.warp_func_st = None;
        # self.warp_func_ts = None;


    def getTargetStart(self, lead=None):
        target_start = self.target_events[0].getStartTime();
        if (lead is None):
            lead = min(target_start, DEFAULT_LEAD_TIME);
        return target_start - lead;


    def getTargetEnd(self, lead=None):
        lastind = min(len(self.source_events), len(self.target_events)) - 1;
        return self.target_events[lastind].getStartTime() + DEFAULT_TAIL_TIME;


    def getSourceStart(self):
        source_start = self.source_events[0].getUnrolledStartTime();
        return source_start;


    def getSourceEnd(self):
        lastind = min(len(self.source_events), len(self.target_events)) - 1;
        return self.source_events[lastind].getUnrolledStartTime();


    def getWarpedSourceStart(self, lead=None):
        source_start = self.source_events[0].getUnrolledStartTime();
        if (lead is None):
            lead = min(source_start, DEFAULT_LEAD_TIME);
        return self.warpSourceTime(source_start - lead);


    def getWarpedSourceEnd(self, tail=None):
        last_event = min(len(self.source_events), len(self.target_events)) - 1;
        source_end = self.source_events[last_event].getUnrolledStartTime();
        if (tail is None):
            tail = DEFAULT_TAIL_TIME;
        source_end = source_end + tail;
        return self.warpSourceTime(source_end);


    def setWarpFunc(self, warp_type, **kwargs):
        if (warp_type == 'square'):
            self.warp_func = [Warp.SquareInterp, Warp.SquareInterp];
        elif (warp_type == 'linear'):
            self.warp_func = [Warp.LinearInterp, Warp.LinearInterp];
        elif (warp_type == 'cubic'):
            self.warp_func = [Warp.CubicInterp, Warp.CubicInterp];
        elif (warp_type == 'quad'):
            self.warp_func = [
                Warp.GetEventWarpFunc(self.source_events, self.target_events, Warp.WFunc_Quadratic(), **kwargs),
                Warp.GetEventWarpFunc(self.target_events, self.source_events, Warp.WFunc_Quadratic(), **kwargs)];
        elif (warp_type == 'mouth'):
            self.warp_func = [
                Warp.GetEventWarpFunc(self.source_events, self.target_events, Warp.WFunc_Mouth(**kwargs), **kwargs),
                Warp.GetEventWarpFunc(self.target_events, self.source_events, Warp.WFunc_Mouth(**kwargs), **kwargs)];
        elif (warp_type == 'weight'):
            self.warp_func = [
                Warp.GetEventWarpFunc(self.source_events, self.target_events,
                                      Warp.WFunc_Weight(use_to_weights=None, **kwargs), **kwargs),
                Warp.GetEventWarpFunc(self.target_events, self.source_events,
                                      Warp.WFunc_Weight(use_to_weights=True, **kwargs), **kwargs)];
        elif (warp_type == 'half_accel'):
            self.warp_func = [
                Warp.GetEventWarpFunc(self.source_events, self.target_events, Warp.WFunc_P(p=0.5),
                                      **kwargs),
                Warp.GetEventWarpFunc(self.target_events, self.source_events, Warp.WFunc_P(p=0.5),
                                      **kwargs)];
        elif (warp_type == 'p'):
            p = kwargs.get('p');
            self.warp_func = [
                Warp.GetEventWarpFunc(self.source_events, self.target_events, Warp.WFunc_P(p=p),
                                      **kwargs),
                Warp.GetEventWarpFunc(self.target_events, self.source_events, Warp.WFunc_P(p=p),
                                      **kwargs)];
        elif (warp_type == 'target_time_source_fraction'):
            self.warp_func = [
                Warp.GetEventWarpFunc(self.source_events, self.target_events,
                                      Warp.WFunc_targettime_sourcefraction(**kwargs),
                                      **kwargs),
                Warp.GetEventWarpFunc(self.target_events, self.source_events,
                                      Warp.WFunc_targettime_sourcefraction(**kwargs),
                                      **kwargs)];
        elif (warp_type == 'target_source_fractions'):
            self.warp_func = [
                Warp.GetEventWarpFunc(self.source_events, self.target_events,
                                      Warp.WFunc_target_source_fractions(**kwargs),
                                      **kwargs),
                Warp.GetEventWarpFunc(self.target_events, self.source_events,
                                      Warp.WFunc_target_source_fractions(**kwargs),
                                      **kwargs)];
        elif (warp_type is not None):
            self.warp_func = [warp_type, warp_type];

        self.setInfo('WarpType', warp_type);
        return;


    def warpSourceTime(self, t):
        return self.warp_func[0](t, a_events=self.source_events, b_events=self.target_events);


    def warpSourceTimes(self, t):
        tw = t.copy();
        for a in range(len(t)):
            tw[a] = self.warpSourceTime(t[a]);
        return tw;


    def warpTargetTime(self, t):
        return self.warp_func[1](t, a_events=self.target_events, b_events=self.source_events);


    def warpTargetTimes(self, t):
        tw = t.copy();
        for a in range(len(t)):
            tw[a] = self.warpTargetTime(t[a]);
        return tw;


    def plot(self, xlim=None, sampling_rate=None, new_figure=None, render_control_points=True, render_labels=True,
             time_range=None, full_source_range=None, **kwargs):
        if (sampling_rate is None):
            sampling_rate = 30;

        source_duration = self.getSourceEnd() - self.getSourceStart();
        old_frame_time = truediv(1.0, sampling_rate);
        target_start = self.getTargetStart();
        target_end = self.getTargetEnd();



        # if(xlim is not None):
        #     target_start=xlim[0]

        target_duration = target_end - target_start;

        new_n_samples = target_duration * sampling_rate;
        target_start_times = np.linspace(target_start, target_end, num=new_n_samples, endpoint=False);

        unwarped_target_times = [];
        for st in target_start_times:
            unwarped_target_times.append(self.warpTargetTime(st));

        if (new_figure):
            fig = plt.figure();

        unwarped_target_times = np.array(unwarped_target_times);
        # unwarped_target_times = unwarped_target_times-unwarped_target_times[0]+target_start;
        plt.plot(target_start_times, unwarped_target_times, '-');

        if (render_control_points):
            lastind = min(len(self.source_events), len(self.target_events)) - 1;
            targeteventtimes = Event.ToStartTimes(self.target_events[:lastind]);
            sourceeventtimes = Event.ToStartTimes(self.source_events[:lastind]);
            plt.plot(targeteventtimes, sourceeventtimes, 'o', label='Control Points');
        if (xlim is not None):
            xrng = [xlim[0] + target_start, xlim[1] + target_start];
            ylim = [self.warpTargetTime(xrng[0]), self.warpTargetTime(xrng[1])];
            plt.xlim(xrng);
            plt.ylim(ylim);

        if (time_range is not None):
            plt.xlim(time_range);

        if (render_labels):
            plt.ylabel('Source Time');
            plt.xlabel('Target Time');

        plt.title('Warp Curve')

        if (new_figure is not None):
            return fig;


    def plotImage(self, xlim=None, sampling_rate=None):
        if (sampling_rate is None):
            sampling_rate = 30;
        target_start = self.getTargetStart();
        target_end = self.getTargetEnd() + 10;
        target_duration = target_end - target_start;

        new_n_samples = target_duration * sampling_rate;
        target_start_times = np.linspace(target_start, target_end, num=new_n_samples, endpoint=True);

        unwarped_target_times = [];
        for st in target_start_times:
            unwarped_target_times.append(self.warpTargetTime(st));

        unwarped_target_times = np.array(unwarped_target_times);
        pim = Image.PlotImage(signal=unwarped_target_times, show_axis=True, xvals=target_start_times,
                              sampling_rate=sampling_rate, events=self.target_events, xlime=[0, 100], ylims=[0, 110]);
        return pim;


    def repeatShorterEvents(self, endpoints=False):
        n_events = max(len(self.source_events), len(self.target_events));
        self.source_events = Event.RepeatToLength(self.source_events, n=n_events, endpoints=endpoints);
        self.target_events = Event.RepeatToLength(self.target_events, n=n_events, endpoints=endpoints);


    @staticmethod
    def FromEvents(source_events, target_events):
        w = Warp();
        w.source_events = source_events;
        w.target_events = target_events;
        # w.repeatShorterEvents();
        return w


    @staticmethod
    def LinearInterp(t, a_events, b_events):
        n_events = min(len(a_events), len(b_events));
        next_a_event_index = n_events;
        for s in range(n_events):
            if (t < a_events[s].start):
                next_a_event_index = s;
                break;

        prev_a_event_time = 0;
        prev_b_event_time = 0;
        if (next_a_event_index > 0):
            prev_a_event_time = a_events[next_a_event_index - 1].start;
            prev_b_event_time = b_events[next_a_event_index - 1].start;

        next_a_event_time = a_events[n_events - 1].start;
        next_b_event_time = b_events[n_events - 1].start;
        if (next_a_event_index < n_events):
            next_a_event_time = a_events[next_a_event_index].start;
            next_b_event_time = b_events[next_a_event_index].start;

        a_event_gap = next_a_event_time - prev_a_event_time;
        # b_event_gap = next_b_event_time - prev_b_event_time;
        t_progress = t - prev_a_event_time;

        # take care of past-the-end case, by simply letting time proceed normally past the last event
        if (a_event_gap == 0):
            return prev_b_event_time + t_progress;

        next_weight = t_progress / (1.0 * a_event_gap);
        return (next_weight * next_b_event_time) + ((1.0 - next_weight) * prev_b_event_time);


    # additional_points = [];
    # for i in range(-10, 11):
    #     additional_points.append([0.1 * i, 0.51])
    @staticmethod
    def plotWarpMethodTest(warp_type, additional_points=None, **kwargs):
        sev = [];
        tev = [];

        pts = [[1.0, 1.0],
               [-1.0, 1.0],
               [0.5, 1.0],
               [1.0, 0.5],
               [-1.0, 0.5],
               [0.5, 0.5],
               [1.0, 0.3],
               [-1.0, 0.3],
               [0.5, 0.3]];

        if (additional_points is not None):
            pts = pts + additional_points;

        currentt = [0.0, 0.0];
        sev.append(Event(start=0.0));
        tev.append(Event(start=0.0));
        for p in pts:
            currentt[0] = currentt[0] + p[0];
            currentt[1] = currentt[1] + p[1];
            sev.append(Event(start=currentt[0]));
            tev.append(Event(start=currentt[1]));

        # warp_type = 'target_time_source_fraction';
        # other_params['acceleration_target_time'] = 0.5;
        # other_params['acceleration_source_fraction'] = 0.75;
        warpf = Warp.FromEvents(sev, tev);
        warpf.setWarpFunc(warp_type, **kwargs);

        warpf.plot()


    @staticmethod
    def SquareInterp(t, a_events, b_events):
        n_events = min(len(a_events), len(b_events));
        next_a_event_index = n_events;
        for s in range(n_events):
            if (t < a_events[s].start):
                next_a_event_index = s;
                break;

        prev_a_event_time = 0;
        prev_b_event_time = 0;
        if (next_a_event_index > 0):
            prev_a_event_time = a_events[next_a_event_index - 1].start;
            prev_b_event_time = b_events[next_a_event_index - 1].start;

        next_a_event_time = a_events[n_events - 1].start;
        next_b_event_time = b_events[n_events - 1].start;
        if (next_a_event_index < n_events):
            next_a_event_time = a_events[next_a_event_index].start;
            next_b_event_time = b_events[next_a_event_index].start;

        a_event_gap = next_a_event_time - prev_a_event_time;
        # b_event_gap = next_b_event_time - prev_b_event_time;
        t_progress = t - prev_a_event_time;

        # take care of past-the-end case, by simply letting time proceed normally past the last event
        if (a_event_gap == 0):
            return prev_b_event_time + t_progress;

        progress_frac = t_progress / (1.0 * a_event_gap);
        next_weight = math.pow(progress_frac, 2);
        # accel = 3;
        # bweight = math.pow(progress_frac,accel);
        # aweight = math.pow(1.0-progress_frac,accel);
        # sumweight = aweight+bweight;
        # next_weight=bweight/sumweight;
        return (next_weight * next_b_event_time) + ((1.0 - next_weight) * prev_b_event_time);


    @staticmethod
    def GetEventWarpFunc(from_events, to_events, f, lead_time=None, **kwargs):
        start_cap_time = min(from_events[0].start, to_events[0].start);
        if (lead_time is not None):
            start_cap_time = min(start_cap_time, lead_time);
        n_events = min(len(from_events), len(to_events));
        # f_events = Event.GetUnrolledList(from_events[:n_events], assert_on_folds=True);
        f_events = Event.GetUnrolledList(from_events[:n_events]);
        t_events = Event.GetUnrolledList(to_events[:n_events]);

        def rfunc(t, **kwargs):
            next_f_event_index = n_events - 1;
            for e in range(n_events):
                if (t < f_events[e].unrolled_start):
                    next_f_event_index = e;
                    break;

            if (next_f_event_index == 0):
                from_cap_event = Event(start=f_events[0].start - start_cap_time, weight=0, type='startcap');
                to_cap_event = Event(start=t_events[0].start - start_cap_time, weight=0, type='startcap')
                return f(t, f_neighbors=[from_cap_event, f_events[0]], t_neighbors=[to_cap_event, t_events[0]]);

            else:
                return f(t,
                         f_neighbors=[f_events[next_f_event_index - 1], f_events[next_f_event_index]],
                         t_neighbors=[t_events[next_f_event_index - 1], t_events[next_f_event_index]]);

        return rfunc;


    @staticmethod
    def WFunc_Quadratic():
        def rfunc(t, f_neighbors, t_neighbors, **kwargs):
            from_times = Event.ToStartTimes(f_neighbors);
            to_times = Event.ToStartTimes(t_neighbors);
            from_event_gap = from_times[1] - from_times[0];
            t_progress = t - from_times[0];
            # avoid divide by 0
            if (from_event_gap == 0):
                AWARN("Event Gap was 0! Check Warp.py")
                return prev_to_event_time + t_progress;
            progress_frac = truediv(t_progress, from_event_gap);

            next_weight = math.pow(progress_frac, 2);

            return (next_weight * to_times[1]) + ((1.0 - next_weight) * to_times[0]);

        return rfunc;


    @staticmethod
    def WFunc_Weight(use_to_weights=None, **kwargs):
        print("USING WEIGHT-BASED WARP");

        def rfunc(t, f_neighbors, t_neighbors):
            from_times = Event.ToStartTimes(f_neighbors);
            to_times = Event.ToStartTimes(t_neighbors);
            from_event_gap = from_times[1] - from_times[0];
            to_event_gap = to_times[1] - to_times[0];
            t_progress = t - from_times[0];
            # avoid divide by 0
            if (from_event_gap == 0):
                AWARN("Event Gap was 0! Check Warp.py")
                return prev_to_event_time + t_progress;
            progress_frac = truediv(t_progress, from_event_gap);

            if (use_to_weights):
                weight = t_neighbors[1].weight;
            else:
                weight = f_neighbors[1].weight;

            p = 1.0 - weight;
            # a = truediv(1.0, (1.0+p*p-2*p)); # if a=b
            a = 1.0 - np.power((1.0 - p), 2.0);  # b=1
            if (progress_frac < p):
                next_weight = a * progress_frac;
            else:
                next_weight = (a * progress_frac) + np.power((progress_frac - p), 2.0);
            return (next_weight * to_times[1]) + ((1.0 - next_weight) * to_times[0]);

        return rfunc;


    @staticmethod
    def WFunc_P(p=None, **kwargs):
        if (p is None):
            p = 0.5;
        print(("USING P WARP with P={}".format(p)));

        def rfunc(t, f_neighbors, t_neighbors):
            from_times = Event.ToStartTimes(f_neighbors);
            to_times = Event.ToStartTimes(t_neighbors);
            from_event_gap = from_times[1] - from_times[0];
            to_event_gap = to_times[1] - to_times[0];
            t_progress = t - from_times[0];
            # avoid divide by 0
            if (from_event_gap == 0):
                AWARN("Event Gap was 0! Check Warp.py")
                return prev_to_event_time + t_progress;
            progress_frac = truediv(t_progress, from_event_gap);
            # a = truediv(1.0, (1.0+p*p-2*p)); # if a=b
            a = 1.0 - np.power((1.0 - p), 2.0);  # b=1
            if (progress_frac < p):
                next_weight = a * progress_frac;
            else:
                next_weight = (a * progress_frac) + np.power((progress_frac - p), 2.0);
            return (next_weight * to_times[1]) + ((1.0 - next_weight) * to_times[0]);

        return rfunc;


    @staticmethod
    def WFunc_Mouth(p_acceleration_time=0.1, **kwargs):
        def rfunc(t, f_neighbors, t_neighbors):
            from_times = Event.ToStartTimes(f_neighbors);
            to_times = Event.ToStartTimes(t_neighbors);
            from_event_gap = from_times[1] - from_times[0];
            to_event_gap = to_times[1] - to_times[0];
            t_progress = t - from_times[0];
            # avoid divide by 0
            if (from_event_gap == 0):
                AWARN("Event Gap was 0! Check Warp.py")
                return prev_to_event_time + t_progress;
            progress_frac = truediv(t_progress, from_event_gap);

            if (f_neighbors[1].type == 'mouth_open' or t_neighbors[1].type == 'mouth_open'):
                p = 1.0 - truediv(p_acceleration_time, to_event_gap);
                if (p < 0):
                    next_weight = math.pow(progress_frac, 2);
                else:
                    # a = truediv(1.0, (1.0+p*p-2*p)); # if a=b
                    a = 1.0 - np.power((1.0 - p), 2.0);  # b=1
                    if (progress_frac < p):
                        next_weight = a * progress_frac;
                    else:
                        next_weight = (a * progress_frac) + np.power((progress_frac - p), 2.0);
            else:
                next_weight = progress_frac;

            return (next_weight * to_times[1]) + ((1.0 - next_weight) * to_times[0]);

        return rfunc;


    @staticmethod
    def WFunc_targettime_sourcefraction(acceleration_target_time=0.1, acceleration_source_fraction=0.8, **kwargs):
        """
        This assumes that you are mapping from the target to the source, as is the most common use case.
        :param acceleration_target_time: amount of from time to spend accelerating
        :param acceleration_source_fraction: fraction of source to accelerate through
        :return:
        """
        lin_source_fraction = 1.0 - acceleration_source_fraction;

        def rfunc(t, f_neighbors, t_neighbors):
            from_times = Event.ToStartTimes(f_neighbors);
            to_times = Event.ToStartTimes(t_neighbors);
            from_event_gap = from_times[1] - from_times[0];
            to_event_gap = to_times[1] - to_times[0];
            t_progress = t - from_times[0];
            # avoid divide by 0
            if (from_event_gap == 0):
                AWARN("Event Gap was 0! Check Warp.py")
                return to_times[
                           0] + t_progress;  # is this right? doesnt seem to come up, not sure what behavior should be looking back on it...

            progress_frac = truediv(t_progress, from_event_gap);

            time_left = from_event_gap - t_progress;

            p = 1.0 - truediv(acceleration_target_time, from_event_gap);
            p2 = p * p;
            q = 1.0 - acceleration_source_fraction;

            if (acceleration_target_time >= from_event_gap or q > (p2)):
                next_weight = math.pow(progress_frac, 2);
            elif (time_left >= acceleration_target_time):
                lin_t_progress_frac = truediv(t_progress, from_event_gap - acceleration_target_time);
                next_weight = lin_t_progress_frac * lin_source_fraction;
            else:
                pdnom = (p2 - 2.0 * p + 1.0)

                # I just used matlab to symbolic inverse matrix of equations, then simplified by hand
                a = ((1.0 - q) / pdnom) + (q / (p * (p - 1)));
                b = ((2 * p * (q - 1)) / pdnom) - ((q * (p + 1)) / (p2 - p))
                # c = ((p2 - (q * (2 * p - 1))) / pdnom) + (q / (p - 1));
                c = 1 - a - b;

                next_weight = a * progress_frac * progress_frac + b * progress_frac + c;
            return (next_weight * to_times[1]) + ((1.0 - next_weight) * to_times[0]);

        return rfunc;


    @staticmethod
    def WFunc_target_source_fractions(acceleration_target_fraction=0.8, acceleration_source_fraction=0.9, **kwargs):
        """
        This assumes that you are mapping from the target to the source, as is the most common use case.
        :param acceleration_target_fraction: fraction of target to spend accelerating
        :param acceleration_source_fraction: fraction of source to accelerate through
        :return:
        """
        lin_source_fraction = 1.0 - acceleration_source_fraction;



        def rfunc(t, f_neighbors, t_neighbors):
            print(acceleration_target_fraction)
            print(acceleration_source_fraction)
            from_times = Event.ToStartTimes(f_neighbors);
            to_times = Event.ToStartTimes(t_neighbors);
            from_event_gap = from_times[1] - from_times[0];
            to_event_gap = to_times[1] - to_times[0];
            t_progress = t - from_times[0];
            # avoid divide by 0
            if (from_event_gap == 0):
                AWARN("Event Gap was 0! Check Warp.py")
                return to_times[
                           0] + t_progress;  # is this right? doesnt seem to come up, not sure what behavior should be looking back on it...

            progress_frac = truediv(t_progress, from_event_gap);

            time_left = from_event_gap - t_progress;

            p = 1.0 - acceleration_target_fraction;
            p2 = p * p;
            q = 1.0 - acceleration_source_fraction;

            if (acceleration_target_fraction >= 1 or q > (p2)):
                next_weight = math.pow(progress_frac, 2);
            elif (progress_frac <= p):
                lin_t_progress_frac = truediv(t_progress, p * from_event_gap);
                next_weight = lin_t_progress_frac * lin_source_fraction;
            else:
                pdnom = (p2 - 2.0 * p + 1.0)

                # I just used matlab to symbolic inverse matrix of equations, then simplified by hand
                a = ((1.0 - q) / pdnom) + (q / (p * (p - 1)));
                b = ((2 * p * (q - 1)) / pdnom) - ((q * (p + 1)) / (p2 - p))
                # c = ((p2 - (q * (2 * p - 1))) / pdnom) + (q / (p - 1));
                c = 1 - a - b;

                next_weight = a * progress_frac * progress_frac + b * progress_frac + c;
            return (next_weight * to_times[1]) + ((1.0 - next_weight) * to_times[0]);

        return rfunc;


    @staticmethod
    def WFunc_AB(const_factor, quad_factor, **kwargs):
        def rfunc(t, f_neighbors, t_neighbors):
            from_times = Event.ToStartTimes(f_neighbors);
            to_times = Event.ToStartTimes(t_neighbors);
            from_event_gap = from_times[1] - from_times[0];
            t_progress = t - from_times[0];
            # avoid divide by 0
            if (from_event_gap == 0):
                AWARN("Event Gap was 0! Check Warp.py")
                return prev_to_event_time + t_progress;
            progress_frac = truediv(t_progress, from_event_gap);

            next_weight = math.pow(progress_frac, 2);

            return (next_weight * to_times[1]) + ((1.0 - next_weight) * to_times[0]);


    @staticmethod
    def ABWarp():
        next_from_event_time = from_events[n_events - 1].start;
        next_to_event_time = to_events[n_events - 1].start;
        if (next_from_event_index < n_events):
            next_from_event_time = from_events[next_from_event_index].start;
            next_to_event_time = to_events[next_to_event_index].start;

        from_event_gap = next_from_event_time - prev_from_event_time;
        # b_event_gap = next_b_event_time - prev_b_event_time;
        t_progress = t - prev_from_event_time;

        # take care of past-the-end case, by simply letting time proceed normally past the last event
        if (from_event_gap == 0):
            return prev_to_event_time + t_progress;

        progress_frac = t_progress / (1.0 * from_event_gap);
        next_weight = math.pow(progress_frac, 2);
        return (next_weight * next_to_event_time) + ((1.0 - next_weight) * prev_to_event_time);


    @staticmethod
    def CubicInterp(t, a_events, b_events):
        # def CubicInterp(a_events, b_events, t):
        f = Warp.CubicInterpFunc(a_events=a_events, b_events=b_events);
        return f(t);


    @staticmethod
    def LinearInterpFunc(a_events, b_events):
        if (a_events[0].start > 0):
            ae = np.concatenate((np.asarray([0]), Event.ToStartTimes(a_events)));
        else:
            ae = Event.ToStartTimes(a_events);
        be = Event.ToStartTimes(b_events);
        if (len(be) > len(ae)):
            be = be[:len(ae)];
        elif (len(ae) > len(be)):
            ae = ae[:len(be)];
        return sp.interpolate.interp1d(ae, be, 'linear', bounds_error=False, fill_value='extrapolate');


    @staticmethod
    def CubicInterpFunc(a_events, b_events):
        event_times = Event.ToStartTimes(a_events);
        wevent_times = Event.ToStartTimes(b_events);
        minlen = min(len(event_times), len(wevent_times));
        event_times = event_times[:minlen];
        wevent_times = wevent_times[:minlen];
        splinecap = np.arange(0, 1, 0.25);

        spline_times = np.concatenate((splinecap + event_times[0] - 1, event_times));
        wspline_times = np.concatenate((splinecap + wevent_times[0] - 1, wevent_times));

        spline_times = np.append(spline_times, splinecap + 0.5 + spline_times[-1]);
        wspline_times = np.append(wspline_times, splinecap + 0.5 + wspline_times[-1]);

        # get spline to translate initial times to warped times
        splineX = sp.interpolate.interp1d(spline_times, wspline_times, kind='cubic', bounds_error=False,
                                          fill_value='extrapolate');
        return splineX;
