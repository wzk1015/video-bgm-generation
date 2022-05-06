from .Event import *

class VisualBeat(Event):
    def VBOBJECT_TYPE(self):
        return 'VisualBeat';

    def __init__(self, start=None, type=None, weight=None, index=None, unrolled_start = None, direction=0):
        Event.__init__(self, start=start, type=type, weight=weight, index=index, unrolled_start=unrolled_start, direction=direction);

    def initializeBlank(self):
        Event.initializeBlank(self);
        self.flow_histogram = None;
        self.local_autocor = None;
        self.sampling_rate = None;

    def __str__(self):
        return "start:{}\ntype:{}\nweight:{}\nindex:{}\nunrolled_start:{}\nis_active:{}\n".format(self.start, self.type, self.weight, self.index, self.unrolled_start, self.is_active);

    # def toGUIDict(self, is_active=1):
    #     return dict(start=self.start, index=self.index, is_active=None)

    def toDictionary(self):
        d=Event.toDictionary(self);
        # d['start']=self.start;
        # d['type']=self.type;
        # d['weight']=self.weight;
        # d['index']=self.index;
        # d['unrolled_start']=self.unrolled_start;
        d['flow_histogram']=self.flow_histogram;
        d['local_autocor']=self.local_autocor;
        d['sampling_rate']=self.sampling_rate;
        return d;

    def initFromDictionary(self, d):
        Event.initFromDictionary(self, d);
        self.start = d['start'];
        self.type = d['type'];
        self.weight = d['weight'];
        self.index = d['index'];
        self.unrolled_start = d['unrolled_start'];
        self.flow_histogram = d.get('flow_histogram');
        self.local_autocor = d.get('local_autocor');
        self.sampling_rate = d.get('sampling_rate');


    def clone(self, start=None):
        if(start):
            newv = VisualBeat(start = start, type=self.type, weight=self.weight, index=self.index);
        else:
            newv = VisualBeat(start = self.start, type=self.type, weight=self.weight, index=self.index);
        newv.flow_histogram = self.flow_histogram.copy();
        newv.local_autocor = self.local_autocor.copy();
        newv.sampling_rate = self.sampling_rate;
        return newv;

    @staticmethod
    def FromEvent(e):
        if(isinstance(e,VisualBeat)):
            return e.clone();
        else:
            return VisualBeat(start=e.start, type=e.type, weight=e.weight, index=e.index);

    @staticmethod
    def time_window_func(max_separation, break_on_cuts=None):
        def window_func(a, b):
            if(break_on_cuts and (a.type=='cut' or b.type=='cut')):
                return False;
            if(np.fabs(a.start-b.start)<max_separation):
                return True;
            else:
                return False;
        return window_func;

    @staticmethod
    def tempo_binary_objective(target_period, binary_weight=None):
        if(binary_weight is None):
            binary_weight = 1.0;
        def objective_func(a, b):
            T = np.fabs(a.start-b.start);
            return -np.power(np.log(truediv(T,target_period)), 2.0)*binary_weight;
        return objective_func;

    @staticmethod
    def autocor_binary_objective(binary_weight=None, **kwargs):
        if (binary_weight is None):
            binary_weight = 1.0;

        def objective_func(a, b):
            T = np.fabs(a.start - b.start);

            # assert(a.sampling_rate), "b has no sampling rate"
            # assert(b.sampling_rate), "a has no sampling rate"
            abin = int(round(np.true_divide(T,a.sampling_rate)));
            score = (a.local_autocor[abin] - 1);
            if (T < 0.25):
                score = -1;
            if(T>3.75):
                score=-1;
            return binary_weight*score;

        return objective_func;

    @staticmethod
    def angle_binary_objective(binary_weight=None, absolute=None):
        if (binary_weight is None):
            binary_weight = 1.0;
        # if (angle_weight is None):
        #     angle_weight = 0.0;

        def objective_func(a, b):
            # T = np.fabs(a.start - b.start);
            # tempo_score = -np.power(np.log(truediv(T, target_period)), 2.0) * binary_weight;
            # should mask out unary contribution from orthogonal angles
            if(absolute):
                return binary_weight * (np.fabs(np.dot(a.flow_histogram, b.flow_histogram)) - 0.70710678118);  # cos 45 degrees
            else:
                return binary_weight * (np.dot(a.flow_histogram, b.flow_histogram));  # cos 45 degrees
        return objective_func;

    @staticmethod
    def Double(events, type=None):
        doubled = [];
        for e in range(1, len(events)):
            halfstart = 0.5 * (events[e].start + events[e - 1].start);
            newhevent = events[e].clone(start=halfstart);
            if(type is not None):
                newhevent.type = type;
            doubled.append(newhevent);
            doubled.append(events[e]);
        return doubled;

    @staticmethod
    def weight_unary_objective(unary_weight=None):
        if(unary_weight is None):
            unary_weight = 1.0;
        def getweight_func(b):
            return unary_weight*b.weight;
        return getweight_func;

    @staticmethod
    def PullOptimalPaths_Basic(vis_beats, target_period, unary_weight=None, binary_weight=None, window_size=None, break_on_cuts = None):
        if(window_size is None):
            window_size = DEFAULT_WINDOW_FACTOR*target_period;
        binary_objective = VisualBeat.tempo_binary_objective(target_period=target_period, binary_weight = binary_weight);
        unary_objective = VisualBeat.weight_unary_objective(unary_weight=unary_weight);
        window_function = VisualBeat.time_window_func(max_separation = window_size, break_on_cuts=break_on_cuts);
        return VisualBeat.DynamicProgramOptimalPaths(vis_beats=vis_beats,
                                                     unary_objective_func=unary_objective,
                                                     binary_objective_func=binary_objective,
                                                     window_func=window_function);

    @staticmethod
    def PullOptimalPaths(vis_beats, unary_fn=None, binary_fn=None, window_fn=None,  target_period=None, unary_weight=None, binary_weight=None, window_size=None,
                               break_on_cuts=None):
        if (window_size is None):
            window_size = DEFAULT_WINDOW_FACTOR * target_period;

        if(binary_fn == 'autocor'):
            binary_objective = VisualBeat.autocor_binary_objective(binary_weight=binary_weight);
        elif(binary_fn == 'angle'):
            binary_objective = VisualBeat.angle_binary_objective(binary_weight=binary_weight);
        else:
            binary_objective = VisualBeat.tempo_binary_objective(target_period=target_period, binary_weight=binary_weight);

        unary_objective = VisualBeat.weight_unary_objective(unary_weight=unary_weight);
        window_function = VisualBeat.time_window_func(max_separation=window_size, break_on_cuts=break_on_cuts);
        return VisualBeat.DynamicProgramOptimalPaths(vis_beats=vis_beats,
                                                     unary_objective_func=unary_objective,
                                                     binary_objective_func=binary_objective,
                                                     window_func=window_function);

    @staticmethod
    def PullOptimalPaths_Autocor(vis_beats, unary_weight=None, binary_weight=None, window_size=None,
                               break_on_cuts=None, **kwargs):
        if (window_size is None):
            # assert(False), 'no window size provided'
            # window_size = DEFAULT_WINDOW_FACTOR * target_period;
            window_size = 200;
            AWARN('NO WINDOWSIZE PROVIDED! PullOptimalPaths_Autocor');
        binary_objective = VisualBeat.autocor_binary_objective(binary_weight=binary_weight);
        unary_objective = VisualBeat.weight_unary_objective(unary_weight=unary_weight);
        window_function = VisualBeat.time_window_func(max_separation=window_size, break_on_cuts=break_on_cuts);
        return VisualBeat.DynamicProgramOptimalPaths(vis_beats=vis_beats,
                                                     unary_objective_func=unary_objective,
                                                     binary_objective_func=binary_objective,
                                                     window_func=window_function);

    @staticmethod
    def DynamicProgramOptimalPaths(vis_beats, unary_objective_func, binary_objective_func, window_func):
        class Node(object):
            def __init__(self, object, prev_node=None):
                self.object = object;
                self.prev_node = prev_node;
                self.cum_score=None;

        nodes = [];
        beats = Event.GetSorted(vis_beats);
        Event.ApplyIndices(beats);

        for b in beats:
            nodes.append(Node(object=b));

        nodes[0].prev_node = None;
        nodes[0].cum_score = unary_objective_func(nodes[0].object);
        current_segment = [];
        segments = [];
        for n in range(1,len(nodes)):
            current_node = nodes[n];
            current_segment.append(current_node);
            options = [];
            j = n-1;
            while(j>=0 and window_func(current_node.object,nodes[j].object)):
                options.append(nodes[j]);
                j = j-1;
            if(len(options)==0):
                current_node.prev_node=None;
                current_node.cum_score=unary_objective_func(current_node.object);
                segments.append(current_segment);
                current_segment = [];
            else:
                best_choice = options[0];
                best_score = options[0].cum_score+binary_objective_func(current_node.object, best_choice.object);
                for o in range(1,len(options)):
                    score = options[o].cum_score+binary_objective_func(current_node.object, options[o].object);
                    if(score>best_score):
                        best_choice=options[o];
                        best_score=score;
                current_node.prev_node = best_choice;
                current_node.cum_score = best_score+unary_objective_func(current_node.object);
        if(len(current_segment)>0):
            segments.append(current_segment);
        sequences = [];
        for S in segments:
            seq = [];
            max_node = S[0];
            max_score = max_node.cum_score;
            for n in range(len(S)):
                if(S[n].cum_score>max_score):
                    max_node = S[n];
                    max_score = max_node.cum_score;

            trace_node = max_node;
            while(trace_node.prev_node is not None):
                seq.append(trace_node.object);
                trace_node=trace_node.prev_node;
            seq.reverse();
            sequences.append(seq);

        return sequences;