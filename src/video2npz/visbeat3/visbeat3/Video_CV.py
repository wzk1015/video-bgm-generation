# from Video import *
from .Image import *
from .Event import *
from .VisualBeat import *
import librosa

FEATURE_FUNCS = {};
VIS_FUNCS = {};
VISVIDEO_FUNCS = {};

FLOW_LOG_EPSILON=1.0;
FLOW_UNIT_GAIN=10000.0;

HISTOGRAM_FRAMES_PER_BEAT = 2;
HISTOGRAM_DOWNSAMPLE_LEVELS = 3;


VB_UPSAMPLE_FACTOR = 1.0;
USING_OPENCV = Image.USING_OPENCV;

if(USING_OPENCV):

    ##########################################################################
    # ################ THESE ARE THE FUNCTIONS TO OVERRIDE! ################ #
    # ################     FOR CUSTOM SALIENCY METRICS!     ################ #
    ##########################################################################

    # You can modify them here. But I would suggest elsewhere in your code
    # just setting Video.localRhythmicSaliencyFunction and
    # Video.visualBeatFunction to whatever you want. See how they are assigned
    # when Video_CV is included in Video.py. -Abe

    def localRhythmicSaliencyFunction(self, **kwargs):
        """
        Change to use different function for local saliency
        """
        return self.getVisibleImpactEnvelope(**kwargs);

    def visualBeatFunction(self, **kwargs):
        """
        Change to use different default strategy for selecting visual beats
        """
        # beat_params = dict(
        #     pre_max_time=0.2,
        #     post_max_time=0.2,
        #     pre_avg_time=0.2,
        #     post_avg_time=0.2,
        #     wait_time=0.1,
        # )
        beat_params = self._getDefaultPeakPickingTimeParams();
        beat_params.update(kwargs);
        return self.getVisibleImpacts(**beat_params);

    # #################### This is how they are called #################### #

    def getLocalRhythmicSaliency(self, force_recompute=False, **kwargs):
        feature_name = 'local_rhythmic_saliency';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = dict();
            params.update(kwargs);
            params.update({'force_recompute':force_recompute});
            result = self.localRhythmicSaliencyFunction(**params);
            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);

    def getVisualBeats(self, force_recompute=False, **kwargs):
        feature_name = 'visual_beats';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = kwargs;
            params.update({'force_recompute': force_recompute});
            result = self.visualBeatFunction(**params);
            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);

    # ##################################################################### #

    ##########################################################################
    # ###################################################################### #
    ##########################################################################



    def cvGetGrayFrame(self, f):
        colorframe = self.getFrame(f).astype(np.uint8);
        return ocv.cvtColor(colorframe, ocv.COLOR_RGB2GRAY);

    def getImageFromFrameGray(self, f):
        frame = self.getImageFromFrame(f);
        frame.RGB2Gray();
        return frame;

    def flow2row(ang, amp, bins, subdivs, n_shifts, density):
        h, w = ang.shape[:2];
        ncells = np.power(4, subdivs);
        nperd = np.power(2, subdivs);

        xw = w-n_shifts[1];
        yw = h-n_shifts[0];

        xcells = np.arange(nperd + 1, dtype=float);
        ycells = np.arange(nperd + 1, dtype=float);
        xcells = np.floor((xcells / (nperd)) * xw);
        ycells = np.floor((ycells / (nperd)) * yw);
        # print(n_shifts)

        ahis = np.zeros([ncells * bins, n_shifts[0]*n_shifts[1]]);

        for dy in range(n_shifts[0]):
            for dx in range(n_shifts[1]):
                ampwin = amp[dy:dy+yw,dx:dx+xw];
                angwin = ang[dy:dy+yw,dx:dx+xw];
                cell_counter = 0;
                for x in range(nperd):
                    for y in range(nperd):
                        ystart=int(ycells[y]);
                        yend = int(ycells[y+1]);
                        xstart = int(xcells[x]);
                        xend = int(xcells[x + 1]);
                        angcell = angwin[ystart:yend, xstart:xend];
                        ampcell = ampwin[ystart:yend, xstart:xend];
                        cahis, cbinbounds = np.histogram(angcell.ravel(), bins=bins, range=(0, 2 * np.pi),
                                                         weights=ampcell.ravel(), density=density);
                        # print("ahis shape: {}\ncahis shape: {}\ndx, dy: {}, {}\ncell_counter: {}\nbins: {}\n".format(ahis.shape, cahis.shape, dx, dy, cell_counter, bins));
                        ahis[cell_counter * bins:(cell_counter + 1) * bins, dx+dy*n_shifts[0]] = cahis;
                        cell_counter = cell_counter + 1;
        return ahis;

    def getFlowFrame(self, frame_index):
        prev_frame = self.cvGetGrayFrame(frame_index-1);
        this_frame = self.vbGetGrayFrame(frame_index);
        return cvDenseFlowFarneback(from_image=prev_frame, to_image=this_frame);

    def getFlowFramePolar(self, frame_index):
        """
        :param self:
        :param frame_index:
        :return: polar where polar[:,:,0] is amplitude, and polar[:,:,1] is angle
        """
        flow = self.getFlowFrame(frame_index);
        fx, fy = flow[:,:,0], flow[:,:,1];
        polar = np.zeros(size(flow));
        polar[:,:,0] = np.sqrt(fx * fx + fy * fy);
        polar[:,:,1] = np.arctan2(fy, fx) + np.pi
        return polar;

    def computeDirectogramPowers(self, bins=None, dead_zone= 0.05, density=None, save_if_computed=True, noise_floor_percentile=None, **kwargs):
        if(bins is None):
            bins = 128;
        if(noise_floor_percentile is None):
            noise_floor_percentile = 20;

        print(("Computing Flow Features with deadzone {}".format(dead_zone)))

        signal_dim = 128;
        m_histvals = np.zeros([signal_dim, self.n_frames(), 3]);

        flow_averages = np.zeros([self.n_frames(), 1]);
        sampling_rate=self.sampling_rate;
        duration = self.getDuration();
        nsamples = sampling_rate*duration;
        # print(sampling_rate, duration)
        frame_start_times = np.linspace(0,duration,num=int(nsamples),endpoint=False);
        frame_index_floats = frame_start_times*self.sampling_rate;

        lastframe = self.cvGetGrayFrame(frame_index_floats[0]);

        start_timer=time.time();
        last_timer=start_timer;
        fcounter=0;
        counter = 0;

        for nf in range(len(frame_index_floats)):
            nextframe= self.cvGetGrayFrame(frame_index_floats[nf]);
            flow = cvDenseFlowFarneback(from_image=lastframe, to_image=nextframe);
            h, w = flow.shape[:2];
            fx, fy = flow[:,:,0], flow[:,:,1];

            # if(filter_median):
            #     fx = fx-np.median(fx.ravel());
            #     fy = fy-np.median(fy.ravel());
            #     assert(False), "SHOULDNT BE FILTERING MEDIAN! VESTIGIAL CODE"

            ang = np.arctan2(fy, fx) + np.pi
            amp = np.sqrt(fx*fx+fy*fy);

            winstarty = int(dead_zone*h);
            winendy = h-winstarty;
            winstartx = int(dead_zone*w);
            winendx = w-winstartx;
            angw = ang[winstarty:winendy, winstartx:winendx];
            ampw = amp[winstarty:winendy, winstartx:winendx];

            mask0 = (ampw>np.percentile(ampw,noise_floor_percentile)).astype(float);
            ahis0, cbinbounds = np.histogram(angw.ravel(), bins=bins, range=(0, 2 * np.pi),
                                             weights=mask0.ravel(), density=density);
            ahis1, cbinbounds1 = np.histogram(angw.ravel(), bins=bins, range=(0, 2 * np.pi),
                                             weights=ampw.ravel(), density=density);
            ahis2, cbinbounds2 = np.histogram(angw.ravel(), bins=bins, range=(0, 2 * np.pi),
                                             weights=np.power(ampw, 2).ravel(), density=density);

            m_histvals[:, counter, 0] = ahis0;
            m_histvals[:, counter, 1] = ahis1;
            m_histvals[:, counter, 2] = ahis2;

            lastframe=nextframe;
            counter+=1;
            fcounter+=1;

            if(not (fcounter%50)):
                if((time.time()-last_timer)>10):
                    last_timer=time.time();
                    print(("{}%% done after {} seconds...".format(100.0*truediv(fcounter,len(frame_index_floats)), last_timer-start_timer)));
        params = dict( bins = bins,
                        deadzone=dead_zone,
                        density=density);
        params.update(kwargs);
        self.setFeature(name='directogram_powers', value=m_histvals, params=params);
        if(save_if_computed):
            self.save(features_to_save=['directogram_powers']);
        return m_histvals;

    ######################### Other Features #############################

    def getDirectogramPowers(self, force_recompute=False, **kwargs):
        feature_name = 'directogram_powers';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            flow_powers = self.computeDirectogramPowers(**kwargs);
        return self.getFeature(feature_name);

    # def getDirectogram(self, bins = None, weights=None, density=None, force_recompute=False, save_if_computed=True, **kwargs):
    def getDirectogram(self, **kwargs):
        feature_name = 'directogram';
        force_recompute = kwargs.get('force_recompute');
        if((not self.hasFeature(feature_name)) or force_recompute):
            flow_powers = self.getFeature('directogram_powers');
            fh = flow_powers[:,:,1];
            self.setFeature(name='directogram', value=fh, params=kwargs);
        return self.getFeature(feature_name);

    def getVisualTempo(self, force_recompute=None, **kwargs):
        feature_name = 'visual_tempo';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            vbe = self.getFeature('local_rhythmic_saliency');
            # assert librosa.__version__ == '0.7.1'
            # result = librosa.beat.beat_track(onset_envelope=vbe, sr=self.sampling_rate, hop_length=1, **kwargs);
            result = librosa.beat.beat_track(onset_envelope=vbe, sr=self.sampling_rate, hop_length=1, **kwargs);
            self.setFeature(name=feature_name, value=result, params=kwargs);
        return self.getFeature(feature_name);

    def getVisualTempogram(self, window_length=None, force_recompute=None, norm_columns = None, **kwargs):
        """

        :param self:
        :param window_length: in seconds
        :param force_recompute:
        :param kwargs:
        :return:
        """
        feature_name = 'visual_tempogram';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            if(window_length is None):
                window_length = DEFAULT_TEMPOGRAM_WINDOW_SECONDS;
            params = kwargs;
            params.update({'force_recompute': force_recompute});
            vbe = self.computeImpactEnvelope(cut_suppression_seconds = None);
            onset_envelope = vbe;
            win_length = int(round(window_length * self.sampling_rate));
            sr = self.sampling_rate;
            hop_length = 1;

            center=kwargs.get('center');
            if(center is None):
                center=True;
            window='hann'
            norm=np.inf;
            ac_window = librosa.filters.get_window(window, win_length, fftbins=True)

            # Center the autocorrelation windows
            n = len(onset_envelope)

            if center:
                onset_envelope = np.pad(onset_envelope, int(win_length // 2),
                                        mode='linear_ramp', end_values=[0, 0])
            # Carve onset envelope into frames
            odf_frame = librosa.util.frame(onset_envelope,
                                   frame_length=win_length,
                                   hop_length=hop_length)
            # Truncate to the length of the original signal
            if center:
                odf_frame = odf_frame[:, :n]

            odf_frame = librosa.util.normalize(odf_frame,axis=0,norm=norm)
            if(norm_columns is None):
                norm_columns = True;

            if(norm_columns):
                # Window, autocorrelate, and normalize
                result = librosa.util.normalize(
                    librosa.core.audio.autocorrelate(odf_frame * ac_window[:, np.newaxis], axis=0), norm=norm, axis=0);
            else:
                result = librosa.core.audio.autocorrelate(odf_frame * ac_window[:, np.newaxis], axis=0);
                result = np.true_divide(result, np.max(result.ravel()));

            tempo_bpms = librosa.tempo_frequencies(result.shape[0], hop_length=hop_length, sr=sr)
            self.setFeature(name='tempogram_bpms', value=tempo_bpms);
            ###########
            self.setFeature(name=feature_name, value=result, params=params);

        return self.getFeature(feature_name);

    def getVisibleImpactEnvelope(self, force_recompute=False, **kwargs):
        feature_name = 'impact_envelope';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = kwargs;
            result = self.computeImpactEnvelope(forward=True, backward = False, **kwargs);
            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);

    def getForwardVisibleImpactEnvelope(self, force_recompute=False, **kwargs):
        """
        Same as getVisibleImpactEnvelope by default.
        :param self:
        :param force_recompute:
        :param kwargs:
        :return:
        """
        feature_name = 'forward_visual_impact_envelope';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = kwargs;
            result = self.computeImpactEnvelope(forward=True, backward = False, **kwargs);
            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);

    def getBackwardVisibleImpactEnvelope(self, force_recompute=False, **kwargs):
        feature_name = 'backward_visual_impact_envelope';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = kwargs;
            result = self.computeImpactEnvelope(forward=False, backward = True, **kwargs);
            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);

    def getBothWayVisibleImpactEnvelope(self, force_recompute=False, **kwargs):
        feature_name = 'both_way_visual_impact_envelope';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = kwargs;
            result = self.computeImpactEnvelope(forward=True, backward = True, **kwargs);
            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);

    def getVisibleImpactEnvelopePowers(self, force_recompute=False, **kwargs):
        feature_name = 'impact_envelope_powers';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = kwargs;
            result0 = self.computeImpactEnvelope(power= 0, **params);
            result = np.zeros([len(result0), 3]);
            result[:, 0] = result0;
            result[:, 1] = self.computeImpactEnvelope(power= 1, **params);
            result[:, 2] = self.computeImpactEnvelope(power=2, **params);
            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);

    def getCutTimes(self):
        return Event.ToStartTimes(self.getCutEvents());

    def getCutEvents(self, force_recompute=False, **kwargs):
        """
        Hacky estimate of cuts in a video
        :param self:
        :param force_recompute:
        :param kwargs:
        :return:
        """
        feature_name = 'cut_events';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = kwargs;
            powers = self.getFeature('directogram_powers');
            im0 = powers[:, :, 0];
            im2 = powers[:, :, 2];
            imc = np.true_divide(im2, 0.05 + im0);

            medsig = np.median(imc, 0);
            medall = np.median(imc);
            cutsig = np.true_divide(medsig, medall)
            cut_detection_ratio = kwargs.get('cut_detection_ratio');
            if(cut_detection_ratio is None):
                cut_detection_ratio=CUT_DETECTION_RATIO;
            clipsig = (cutsig > cut_detection_ratio).astype(float);
            clip_floorsig = (cutsig > CUT_DETECTION_FLOOR).astype(float);
            clipsig = np.multiply(clipsig, clip_floorsig);

            einds = np.flatnonzero(clipsig);
            etimes = einds * truediv(1.0, self.sampling_rate);
            ev = Event.FromStartTimes(etimes, type='cut');
            ev = self.visualBeatsFromEvents(ev);
            evout = []
            for e in range(len(ev)):
                ev[e].setInfo('frame', einds[e]);
            result = ev;
            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);

    def visualBeatsFromEvents(self, events):
        def downsample_hist(sig, levels):
            nperbin = np.power(2, levels);
            rshp = sig.reshape(-1, nperbin);
            return rshp.sum(axis=1);
        if(self.hasFeature('impact_envelope')):
            svbe=self.getFeature('impact_envelope');
        else:
            svbe=self.computeImpactEnvelope(cut_suppression_seconds = None);
        flow_powers = self.getFeature('directogram_powers');
        vis_tempogram = self.getFeature('visual_tempogram');

        vbeats = [];
        for e in events:
            b = VisualBeat.FromEvent(e);
            ei = int(round(b.start*self.sampling_rate*VB_UPSAMPLE_FACTOR));
            b.weight = svbe[ei];

            histsize = int(128/int(np.power(2,HISTOGRAM_DOWNSAMPLE_LEVELS)))
            histslice = np.zeros([histsize, HISTOGRAM_FRAMES_PER_BEAT]);
            histslice = np.squeeze(np.mean(histslice, 1));
            b.flow_histogram = downsample_hist(sig=histslice, levels=HISTOGRAM_DOWNSAMPLE_LEVELS);
            b.flow_histogram = np.divide(b.flow_histogram, np.sum(b.flow_histogram));
            b.local_autocor = vis_tempogram[:,ei];
            b.local_autocor = b.local_autocor/np.max(b.local_autocor);
            b.sampling_rate=self.sampling_rate;
            vbeats.append(b);
        return vbeats;

    def getVisualBeatTimes(self, **kwargs):
        return Event.ToStartTimes(self.getVisualBeats(**kwargs));

    def getDirectionalFlux(self,
                            f_sigma=None,
                            median_kernel=None,
                            power=None,
                            **kwargs):
        """
        The visual impact complement of a spectral flux matrix
        :param self:
        :param f_sigma: sigma for the gaussian used to filter, using sp.ndimage.filters.gaussian_filter
        :param median_kernel: used in sp.signal.medfilt
        :param power: Which power of the flow to use. Usually use 1.
        :param kwargs:
        :return:
        """

        def d_x(im):
            d_im = np.zeros(im.shape);
            d_im[:, 0] = im[:, 0];
            for c in range(1, im.shape[1]):
                d_im[:, c] = im[:, c] - im[:, c - 1];
            return d_im;

        if (f_sigma is None):
            f_sigma = [5, 3];
        if (median_kernel is None):
            median_kernel = [3, 3];

        if(power is None):
            power = 1;

        powers = self.getFeature('directogram_powers');
        im = powers[:,:,power].copy();
        if (f_sigma is not None):
            im = sp.ndimage.filters.gaussian_filter(input=im, sigma=f_sigma, order=0);

        im = sp.signal.medfilt(im, median_kernel);
        return d_x(im);

    def computeImpactEnvelope(self,
                             forward=True,
                             backward = False,
                             f_sigma=None,
                             median_kernel=None,
                             highpass_window_seconds= 0.8,
                             cut_percentile=99,
                             power=None,
                             crop = None,
                             normalize = True,
                             **kwargs):
        """

        :param self:
        :param forward: include impact going forward in time
        :param backward: include impact going backward in time
        :param f_sigma: sigma for the gaussian used to filter, using sp.ndimage.filters.gaussian_filter
        :param median_kernel: used in sp.signal.medfilt
        :param highpass_window_seconds: highpass window size in seconds
        :param cut_percentile: percentile above which to consider cuts and to clip
        :param power: Which power of the flow to use. Usually use 1.
        :param crop:
        :param kwargs:
        :return:
        """

        upsample_factor = kwargs.get('upsample_factor');
        inputargs = dict(f_sigma=f_sigma,
                         median_kernel=median_kernel,
                         power=power,
                         crop = crop);

        inputargs.update(kwargs);
        im_d = self.getDirectionalFlux(**inputargs);

        if(forward and backward):
            im = np.fabs(im_d);
        elif(forward):
            im = -im_d;
            im = np.clip(im, 0, None)
        elif(backward):
            im = im_d;
            im = np.clip(im, 0, None)
        else:
            assert(False), "Must be at least one of either forward or backward."

        vimpact = np.squeeze(np.mean(im, 0));
        sampling_rate = self.sampling_rate;
        if(upsample_factor is not None and (upsample_factor>1)):
            newlen = upsample_factor * len(vimpact);
            sampling_rate = upsample_factor*sampling_rate;
            vimpact = sp.signal.resample(vimpact, newlen);

        if(highpass_window_seconds):
            order = kwargs.get('highpass_order');
            if(order is None):
                order = 5;
            cutoff = truediv(1.0, highpass_window_seconds);
            normal_cutoff = cutoff / (sampling_rate*0.5);
            b, a = sp.signal.butter(order, normal_cutoff, btype='high', analog=False)
            vimpact = sp.signal.filtfilt(b, a, vimpact);


        normfactor = np.max(np.fabs(vimpact[:]));

        if (cut_percentile is not None):
            fx = np.fabs(vimpact);
            pv = np.percentile(fx, cut_percentile);
            pvlow = np.percentile(fx, cut_percentile-1);
            normfactor = pv;
            ptile = (vimpact > pv).astype(float);
            pntile = (vimpact < -pv).astype(float);
            pboth = ptile+pntile;
            einds = np.flatnonzero(pboth);
            lastind = -2;
            for j in range(len(einds)):
                if(einds[j]==(lastind+1)):
                    vimpact[einds[j]]=0;
                else:
                    vimpact[einds[j]]=pvlow;

        if(normalize):
            vimpact = np.true_divide(vimpact, normfactor);
        return vimpact;

    # 0.8, cut_suppression_seconds = 0.4,
    def computeImpactEnvelopeOld(self, f_sigma=None, median_kernel=None, highpass_window_seconds= 0.8, cut_percentile=99, power=None, crop = None, normalize=None, **kwargs):
        """
        I believe this is the version of the function that I used in the original paper. Keeping it around for record.
        :param self:
        :param f_sigma:
        :param median_kernel:
        :param highpass_window_seconds:
        :param cut_percentile:
        :param power:
        :param crop:
        :param normalize:
        :param kwargs:
        :return:
        """
        def d_x(im):
            # d_im = np.zeros(im.shape, dtype=np.float128);
            d_im = np.zeros(im.shape);
            d_im[:, 0] = im[:, 0];
            for c in range(1, im.shape[1]):
                d_im[:, c] = im[:, c] - im[:, c - 1];
            return d_im;

        if (f_sigma is None):
            f_sigma = [5, 3];
        if (median_kernel is None):
            median_kernel = [3, 3];

        if(power is None):
            power = 1;

        upsample_factor = kwargs.get('upsample_factor');

        powers = self.getFeature('directogram_powers');
        print(("computing impact env with power {}".format(power)));
        im = powers[:,:,power].copy();

        if (f_sigma is not None):
            im = sp.ndimage.filters.gaussian_filter(input=im, sigma=f_sigma, order=0);

        im = sp.signal.medfilt(im, median_kernel);
        im = -d_x(im);
        im = np.clip(im, 0, None)

        svbe = np.squeeze(np.mean(im, 0));
        sampling_rate = self.sampling_rate;

        if(upsample_factor is not None and (upsample_factor>1)):
            newlen = upsample_factor * len(svbe);
            sampling_rate = upsample_factor*sampling_rate;
            svbe = sp.signal.resample(svbe, newlen);

        if(highpass_window_seconds):
            order = kwargs.get('highpass_order');
            if(order is None):
                order = 5;
            cutoff = truediv(1.0, highpass_window_seconds);
            normal_cutoff = cutoff / (sampling_rate*0.5);
            b, a = sp.signal.butter(order, normal_cutoff, btype='high', analog=False)
            svbe = sp.signal.filtfilt(b, a, svbe);


        normfactor = np.max(np.fabs(svbe[:]));

        if (cut_percentile is not None):
            fx = np.fabs(svbe);
            pv = np.percentile(fx, cut_percentile);
            pvlow = np.percentile(fx, cut_percentile-1);
            normfactor = pv;
            ptile = (svbe > pv).astype(float);
            pntile = (svbe < -pv).astype(float);
            pboth = ptile+pntile;
            einds = np.flatnonzero(pboth);
            lastind = -2;
            for j in range(len(einds)):
                if(einds[j]==(lastind+1)):
                    svbe[einds[j]]=0;
                else:
                    svbe[einds[j]]=pvlow;

        if(normalize is not False):
            svbe = np.true_divide(svbe, normfactor);
        return svbe;

    def getVisibleImpacts(self, force_recompute=False, include_cut_events = None, **kwargs):
        feature_name = 'visible_impacts';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = kwargs;
            svbe = self.getFeature('impact_envelope', **kwargs);
            upsample_factor = kwargs.get('upsample_factor');
            if(upsample_factor is None):
                upsample_factor = 1;
            u_sampling_rate = self.sampling_rate*upsample_factor;

            peak_params = self._getDefaultPeakPickingTimeParams();
            peak_params.update(kwargs); # if params given in arguments, those will override the defaults here.

            v_events = Event.FromSignalPeaks(signal=svbe, sampling_rate=u_sampling_rate, **peak_params);
            if(include_cut_events):
                cut_events = self.getFeature('cut_events');
                v_events = v_events + cut_events;
                Event.Sort(v_events);
            result = self.visualBeatsFromEvents(v_events);

            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);

    def getForwardVisibleImpacts(self, force_recompute=False, include_cut_events = None, **kwargs):
        feature_name = 'forward_visual_beats';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = kwargs;
            local_saliency = self.getFeature('forward_visual_impact_envelope', **kwargs);
            upsample_factor = kwargs.get('upsample_factor');
            if(upsample_factor is None):
                upsample_factor = 1;
            u_sampling_rate = self.sampling_rate*upsample_factor;
            v_events = Event.FromSignalPeaks(signal=local_saliency, sampling_rate=u_sampling_rate, event_type= 'forward', **kwargs);
            if(include_cut_events):
                cut_events = self.getFeature('cut_events');
                v_events = v_events + cut_events;
                Event.Sort(v_events);
            result = Event.SetDirections(v_events, direction=Event.DIRECTION_FORWARD);
            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);

    def getBackwardVisibleImpacts(self, force_recompute=False, include_cut_events = None, **kwargs):
        feature_name = 'backward_visual_beats';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = kwargs;
            local_saliency = self.getFeature('backward_visual_impact_envelope', **kwargs);
            upsample_factor = kwargs.get('upsample_factor');
            if(upsample_factor is None):
                upsample_factor = 1;
            u_sampling_rate = self.sampling_rate*upsample_factor;
            v_events = Event.FromSignalPeaks(signal=local_saliency, sampling_rate=u_sampling_rate, **kwargs);
            if(include_cut_events):
                cut_events = self.getFeature('cut_events');
                v_events = v_events + cut_events;
                Event.Sort(v_events);
            # result = self.visualBeatsFromEvents(v_events);
            result = Event.SetDirections(v_events, direction=Event.DIRECTION_BACKWARD);
            self.setFeature(name=feature_name, value=result, params=params);
        return self.getFeature(feature_name);


    def findAccidentalDanceSequences(self, target_n_beats = 7, n_samples=25, delta_range = None):
        if(delta_range is None):
            delta_range = [0.02, 0.5];

        deltas = np.linspace(delta_range[0], delta_range[1], num=n_samples, endpoint=True);
        deltapick = delta_range[0];
        sequences = [];
        for i, delta in enumerate(deltas):
            peak_vars = self._getDefaultPeakPickingTimeParams(delta=delta);
            sequences = self.getVisualBeatSequences(peak_vars=peak_vars, print_summary=False);
            # print("Delta {} has top sequence with {} beats".format(delta, len(sequences[0])));
            if(len(sequences[0])<=target_n_beats):
                deltapick = delta;
                break;
        print(("Selected delta value {}".format(deltapick)));
        return sequences;


    def getVisualBeatSequences(self,
                               search_tempo=None,
                               target_period=None,
                               search_window=0.75,
                               min_beat_limit=None,
                               max_beat_limit=None,
                               unary_weight=None,
                               binary_weight=None,
                               break_on_cuts=None,
                               peak_vars=None,
                               time_range=None,
                               n_return = None,
                               unsorted = False,
                               print_summary = True,
                               **kwargs):
        """

        :param self:
        :param search_tempo: optional tempo you would like to find visual beats at
        :param target_period: optional target period you would like to use for finding beats. Ignored if search tempo is provided.
        :param search_window: longest amount of time (seconds) allowed between beats before a segment is broken into multiple segments
        :param min_beat_limit: only consider sequences with
        :param unary_weight:
        :param binary_weight:
        :param break_on_cuts:
        :param peak_vars:
        :param kwargs:
        :return:
        """
        if (peak_vars is not None):
            # impacts = self.getFeature('visible_impacts', force_recompute=True, **peak_vars);
            impacts = self.getVisualBeats(force_recompute = True, **peak_vars);
        else:
            # impacts = self.getFeature('visible_impacts');
            impacts = self.getVisualBeats();

        if(time_range is not None):
            impactseg = [];
            for i in impacts:
                if(i.start>time_range[0] and i.start < time_range[1]):
                    impactseg.append(i);
            impacts = impactseg;


        if (search_tempo is not None):
            tempo = search_tempo;
            beat_time = np.true_divide(60.0, tempo);
            sequences = VisualBeat.PullOptimalPaths_Basic(impacts, target_period=beat_time, unary_weight=unary_weight,
                                                      binary_weight=binary_weight, break_on_cuts=break_on_cuts,
                                                      window_size=search_window);

        elif(target_period is not None):
            sequences = VisualBeat.PullOptimalPaths_Basic(impacts, target_period=target_period, unary_weight=unary_weight,
                                                          binary_weight=binary_weight, break_on_cuts=break_on_cuts,
                                                          window_size=search_window);
        else:
            sequences = VisualBeat.PullOptimalPaths_Autocor(impacts, unary_weight=unary_weight, binary_weight=binary_weight,
                                                        break_on_cuts=break_on_cuts, window_size=search_window);

        r_sequences = [];

        if(min_beat_limit is None):
            min_beat_limit = 2;
        if(max_beat_limit is None):
            max_beat_limit = len(impacts)+1;

        for S in sequences:
            if ((len(S) > min_beat_limit) and (len(S) < max_beat_limit)):
                r_sequences.append(S);

        if(not unsorted):
            r_sequences.sort(key=len, reverse=True);
            if(n_return is not None):
                r_sequences = r_sequences[:n_return];
        if(print_summary):
            print(("{} segments".format(len(r_sequences))));
            for s in range(len(r_sequences)):
                print(("Segment {} has {} beats".format(s, len(r_sequences[s]))));

        return r_sequences;


    def printVisualBeatSequences(self,
                                 search_tempo=None,
                                 target_period=None,
                                 search_window=None,
                                 min_beat_limit=None,
                                 max_beat_limit=None,
                                 unary_weight=None,
                                 binary_weight=None,
                                 break_on_cuts=None,
                                 peak_vars=None,
                                 n_return = None,
                                 time_range=None, **kwargs):
        """

        :param self:
        :param target_period:
        :param search_tempo:
        :param search_window:
        :param beat_limit:
        :param unary_weight:
        :param binary_weight:
        :param break_on_cuts:
        :param n_return:
        :param peak_vars:
        :param time_range:
        :param kwargs:
        :return:
        """

        sorted = True;

        sequence_args = dict(
            search_tempo=search_tempo,
            target_period=target_period,
            search_window=search_window,
            min_beat_limit=min_beat_limit,
            max_beat_limit=max_beat_limit,
            unary_weight=unary_weight,
            binary_weight=binary_weight,
            break_on_cuts=break_on_cuts,
            peak_vars=peak_vars,
            n_return=n_return,
            time_range=time_range);

        seqs = self.getVisualBeatSequences(**sequence_args)
        print(("sequence arguments were:\n{}".format(sequence_args)));
        print(("There were {} sequences total".format(len(seqs))));
        nclips = 0;
        rsegments = [];
        for S in seqs:
            if (len(S) > 1):
                nclips = nclips + 1;
                rsegments.append(S);

        # rsegments.sort(key=len, reverse=True);
        # if (n_return is not None):
        #     rsegments = rsegments[:n_return];

        Event.PlotSignalAndEvents(self.getFeature('impact_envelope'),  sampling_rate=self.sampling_rate*VB_UPSAMPLE_FACTOR, events=rsegments[0],  time_range=time_range);
        return rsegments;

    def plotEvents(self, events, time_range = 'default', **kwargs):
        time_range_use = time_range;
        if(time_range.lower() == 'default'):
            time_range_use = [0,0];
            time_range_use[0] = events[0].start-1;
            time_range_use[1] = events[-1].start+1;

        signal = self.getFeature('local_rhythmic_saliency');
        mplt = Event.PlotSignalAndEvents(signal, sampling_rate=self.sampling_rate * VB_UPSAMPLE_FACTOR, events=events, time_range=time_range_use, **kwargs);
        plt.xlabel('Time (s)')
        return mplt;

    def plotCutEvents(self, **kwargs):
        signal = self.getFeature('impact_envelope');
        events = self.getFeature('cut_events', **kwargs);
        Event.PlotSignalAndEvents(signal, sampling_rate=self.sampling_rate*VB_UPSAMPLE_FACTOR, events=events, **kwargs);

    def plotVisibleImpacts(self, **kwargs):
        signal = self.getFeature('impact_envelope');
        events = self.getFeature('visible_impacts', **kwargs);
        Event.PlotSignalAndEvents(signal, sampling_rate=self.sampling_rate*VB_UPSAMPLE_FACTOR, events=events, **kwargs);
        plt.title('Impact Envelope & Visual Beats')
        plt.xlabel('Time (s)')
        plt.ylabel('Impact Strength')

    def plotImpactEnvelope(self, **kwargs):
        signal = self.getFeature('local_rhythmic_saliency');
        # events = self.getFeature('visual_beats', **kwargs);
        events = None;
        Event.PlotSignalAndEvents(signal, sampling_rate=self.sampling_rate*VB_UPSAMPLE_FACTOR, events=events, **kwargs);
        plt.title('Impact Envelope & Visual Beats')
        plt.xlabel('Time (s)')
        plt.ylabel('Impact Strength')

    def plotVisualBeats(self, **kwargs):
        signal = self.getFeature('local_rhythmic_saliency');
        events = self.getFeature('visual_beats', **kwargs);
        Event.PlotSignalAndEvents(signal, sampling_rate=self.sampling_rate*VB_UPSAMPLE_FACTOR, events=events, **kwargs);
        plt.title('Impact Envelope & Visual Beats')
        plt.xlabel('Time (s)')
        plt.ylabel('Impact Strength')


    def loadFlowFeatures(self):
        self.load(features_to_load=['directogram_powers', 'directogram']);

    FEATURE_FUNCS['local_rhythmic_saliency'] = getLocalRhythmicSaliency;
    FEATURE_FUNCS['directogram_powers'] = getDirectogramPowers;
    FEATURE_FUNCS['directogram'] = getDirectogram;
    FEATURE_FUNCS['impact_envelope'] = getVisibleImpactEnvelope;
    FEATURE_FUNCS['impact_envelope_powers'] = getVisibleImpactEnvelopePowers;
    FEATURE_FUNCS['visible_impacts'] = getVisibleImpacts;
    FEATURE_FUNCS['visual_beats'] = getVisualBeats;
    # FEATURE_FUNCS['backward_visual_beats'] = getBackwardVisualBeats;
    # FEATURE_FUNCS['forward_visual_beats'] = getForwardVisualBeats;
    FEATURE_FUNCS['backward_visual_impact_envelope'] = getBackwardVisibleImpactEnvelope;
    FEATURE_FUNCS['both_way_visual_impact_envelope'] = getBothWayVisibleImpactEnvelope;
    FEATURE_FUNCS['forward_visual_impact_envelope'] = getForwardVisibleImpactEnvelope;
    FEATURE_FUNCS['directional_flux'] = getDirectionalFlux;
    FEATURE_FUNCS['visual_tempogram'] = getVisualTempogram;
    FEATURE_FUNCS['cut_events']=getCutEvents;


