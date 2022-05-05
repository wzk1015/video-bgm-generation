#VideoVersion#from VisBeatImports import *
from .AObject import *
from .TimeSignal import *
from .Audio import *
from .Warp import *
from .Image import *
import moviepy.editor as mpy
from moviepy.audio.AudioClip import AudioArrayClip as MPYAudioArrayClip

import sys

import math
from operator import truediv


def MPYWriteVideoFile(mpyclip, filename, **kwargs):
    temp_audio_filename = get_temp_file_path(final_file_path='TEMP_'+filename+'.m4a', temp_dir_path=Video.VIDEO_TEMP_DIR);
    return mpyclip.write_videofile(filename=filename, temp_audiofile= temp_audio_filename, audio_codec='aac', **kwargs);



class Video(TimeSignal):
    """Video (class): A video, and a bunch of convenience functions to go with it.
        Attributes:
    """

    VIDEO_TEMP_DIR = './'
    FEATURE_FUNCS = TimeSignal.FEATURE_FUNCS.copy();

    def AOBJECT_TYPE(self):
        return 'Video';

    def __init__(self, path=None, name=None, num_frames_total=None):
        TimeSignal.__init__(self, path=path);
        if(name):
            self.name = name;
        if(path):
            self.loadFile(num_frames_total=num_frames_total);

    def initializeBlank(self):
        TimeSignal.initializeBlank(self);#YES KEEP
        self.name = None;
        self.sampling_rate = None;
        self.audio=None;
        self.reader = None;
        self.writer = None;
        self.num_frames_total=None;
        self.reshape=None;
        self.meta_data = None;
        self.sampling_rate = None;
        self.source = None;

        # _gui is not saved
        self._gui = None;

    def _getFrameRate(self):
        return self.sampling_rate;

    # _gui is from TimeSignal
    # <editor-fold desc="Property: 'gui'">
    @property
    def gui(self):
        return self._getGui();

    def _getGui(self):
        return self._gui;

    @gui.setter
    def gui(self, value):
        self._setGui(value);

    def _setGui(self, value):
        self._gui = value;
    # </editor-fold>


    def getVersionInfo(self):
        """This is the info to be saved in asset version dictionary"""
        d={};
        d['name']=self.name;
        d['sampling_rate']=self.sampling_rate;
        d['num_frames_total']=self.num_frames_total;
        d['duration']=self.getDuration();
        d['start_time']=self.getStartTime();
        d['end_time']=self.getEndTime();
        d['meta_data']=self.meta_data;
        return d;

    def getName(self):
        if(self.name is None):
            return self.getInfo('file_name');
        else:
            return self.name;

    def getTempDir(self):
        if(self.source is not None):
            return self.source.getDir('temp');
        else:
            return Video.VIDEO_TEMP_DIR;

    def getStringForHTMLStreamingBase64(self):
        svideo = io.open(self.getPath(), 'r+b').read()
        encoded = base64.b64encode(svideo)
        return "data:video/mp4;base64,{0}".format(encoded.decode('ascii'));


    def n_frames(self):
        if(not self.num_frames_total):
            self.num_frames_total = self.calcNumFramesTotal();
        return self.num_frames_total;

    def getDuration(self):
        return truediv(self.n_frames(), self.sampling_rate);

    def getStartTime(self):
        return 0;

    def getEndTime(self):
        return self.getDuration();

    def getMPYClip(self, get_audio=True):
        return mpy.VideoFileClip(self.getPath(), audio=get_audio);

    def getAudio(self):
        return self.audio;

    def loadFile(self, file_path=None, num_frames_total=None):
        if (file_path):
            self.setPath(file_path=file_path);
        if ('file_path' in self.a_info):
            self.reader = imageio.get_reader(self.a_info['file_path'], 'ffmpeg');
            self.meta_data = self.reader.get_meta_data();
            self.sampling_rate = self.meta_data['fps'];
            if (num_frames_total is not None):
                self.num_frames_total = num_frames_total;
            else:
                self.num_frames_total = self.calcNumFramesTotal();

            try:
                self.audio = Audio(self.a_info['file_path']);
                self.audio.name =self.name;
            #except RuntimeError:
            except Exception:
                print(("Issue loading audio for {}".format(self.a_info['file_path'].encode('utf-8'))));
                self.audio = Audio(sampling_rate=16000);
                self.audio.x = np.zeros(int(np.ceil(self.audio.sampling_rate*self.getDuration())));

    def openVideoWriter(self, output_file_path, fps=None):
        if('outputs' not in self.a_info):
            self.a_info['outputs'] = [];
        out_fps = fps;
        if(not out_fps):
            out_fps=self.sampling_rate;
        make_sure_dir_exists(output_file_path);
        self.writer = imageio.get_writer(output_file_path, 'ffmpeg', macro_block_size = None, fps = out_fps);
        self.a_info['outputs'].append(output_file_path);

    def closeVideoWriter(self):
        self.writer.close();
        self.writer = None;

    def getFrameShape(self):
        fs=self.getInfo('frame_shape');
        if(fs is not None):
            return fs;
        else:
            self.setInfo(label='frame_shape', value=self.reader.get_data(0).shape);
            return self.getInfo('frame_shape');

    def calcNumFramesTotal(self):
        #assert(False)
        print(("Calculating frames for {}...".format(self.name)))
        valid_frames = 0
        example_frame = self.reader.get_data(0);
        self.setInfo(label='frame_shape',value=example_frame.shape);
        
        # https://stackoverflow.com/questions/54778001/how-to-to-tackle-overflowerror-cannot-convert-float-infinity-to-integer
        #for i in range(1, self.reader.get_length()):
        for i in range(1, self.reader.count_frames()):
            try:
                self.reader.get_data(i);
            except imageio.core.format.CannotReadFrameError as e:
                break
            valid_frames += 1
        print("Done.")
        return valid_frames

    def readFrameBasic(self, i):
        """You should basically never call this"""
        fi=i;
        if(fi<0):
            fi=0;
        if(fi>(self.num_frames_total-1)):
            fi=(self.num_frames_total-1);
        return np.asarray(self.reader.get_data(int(fi)));

    def getFrame(self, f):
        return self.readFrameBasic(round(f));

    def getFrameFromTime(self, t):
        f = t*self.sampling_rate;
        return self.getFrame(f=f);

    def getFrameLinearInterp(self, f):
        if(isinstance(f,int) or f.is_integer()):
            return self.readFrameBasic(int(f));
        prev_frame = self.readFrameBasic(math.floor(f));
        next_frame = self.readFrameBasic(math.ceil(f));
        frame_progress = f-np.floor(f);
        rframe =  (next_frame*frame_progress)+(prev_frame*(1.0-frame_progress));
        return rframe.astype(prev_frame.dtype);

    def writeFrame(self, img):
        if self.writer.closed:
            print('ERROR: Vid writer object is closed.')
        else:
            self.writer.append_data(img.astype(np.uint8))

    def play(self):
        if(ISNOTEBOOK):
            print("Playing video:")
            video = io.open(self.getPath(), 'r+b').read()
            encoded = base64.b64encode(video)
            vidhtml=HTML(data='''<video alt="test" controls>
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii')));
            IPython.display.display(vidhtml);
            # return vidhtml;
        else:
             print("HOW TO PLAY VIDEO? NOT A NOTEBOOK.")

    def show(self):
        self.play();

    def write(self, output_path, output_sampling_rate=None):
        assert output_path, "MUST PROVIDE OUTPUT PATH FOR VIDEO"

        sampling_rate = output_sampling_rate;
        if(not sampling_rate):
            sampling_rate=self.sampling_rate;
        tempfilepath = get_temp_file_path(final_file_path=output_path, temp_dir_path=self.getTempDir());
        self.openVideoWriter(output_file_path=tempfilepath, fps=output_sampling_rate);

        duration = self.getDuration();
        nsamples = sampling_rate*duration;
        old_frame_time = truediv(1.0,self.sampling_rate);
        frame_start_times = np.linspace(0,self.getDuration(),num=nsamples,endpoint=False);
        frame_index_floats = frame_start_times*self.sampling_rate;

        start_timer=time.time();
        last_timer=start_timer;
        fcounter=0;
        for nf in range(len(frame_index_floats)):
            self.writeFrame(self.getFrame(frame_index_floats[nf]));
            fcounter+=1;
            if(not (fcounter%50)):
                if((time.time()-last_timer)>10):
                    last_timer=time.time();
                    print(("{}%% done after {} seconds...".format(100.0*truediv(fcounter,len(frame_index_floats)), last_timer-start_timer)));


        self.closeVideoWriter();
        rvid = Video.CreateFromVideoAndAudio(video_path=tempfilepath, audio_object=self.audio, output_path=output_path);
        os.remove(tempfilepath);
        return rvid;

    def VideoClip(self, start=None, end=None, name=None):
        from . import VideoClip
        if(start is None):
            start = 0;
        if(end is None):
            end = self.getDuration();
        clip = VideoClip.VideoClip(video=self, start=start, end=end);
        if(name):
            clip.name=name;
        return clip;

    def getImageFromFrame(self, i):
        rimage = Image(data=self.getFrame(i));
        rimage.setInfo(label='parent_video', value=self.getInfo('file_path'));
        rimage.setInfo(label='frame_number',value=i);
        return rimage;

    def getImageFromTime(self, t):
        rimage = Image(data=self.getFrameFromTime(t));
        rimage.setInfo(label='parent_video', value=self.getInfo('file_path'));
        rimage.setInfo(label='frame_time', value=t);
        return rimage;


    # def getFrameShape(self):
    #     return self.getImageFromFrame(0).getShape();

    def writeResolutionCopyFFMPEG(self, path, max_height=None):
        if(max_height is None):
            max_height=self.getFrameShape()[0];
        mpc = self.getMPYClip();
        clip_resized = mpc.resize(height=max_height); # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
        if((clip_resized.size[0]%2)>0):
            clip_resized=clip_resized.crop(x1=0,width=clip_resized.size[0]-1);

        # clip_resized.write_videofile(path);
        MPYWriteVideoFile(clip_resized, path);

        rvid = Video(path);
        return rvid;

    def writeFFMPEG(self, output_path):
        mpc = self.getMPYClip();
        # mpc.write_videofile(output_path, preset='fast', codec='mpeg4');
        MPYWriteVideoFile(mpc, output_path, preset='fast', codec='mpeg4')
        rvid = Video(output_path);
        return rvid;

    def writeResolutionCopy(self, path, max_height=None, reshape=None, input_sampling_rate_factor = None, output_sampling_rate=None):
        print((self.getPath()))
        if(input_sampling_rate_factor is not None):
            original_sampling_rate = self.sampling_rate;
            self.sampling_rate=input_sampling_rate_factor*self.sampling_rate;
            original_audio_sampling_rate = self.audio.sampling_rate;
            self.audio.sampling_rate=self.audio.sampling_rate*input_sampling_rate_factor;
        output_path = path;
        inshape = self.getFrameShape();
        if(reshape==True):
            imshape = [inshape[1],inshape[0],inshape[2]];
        outshape = None;
        if(max_height and (max_height<inshape[0])):
            outshape = self.getFrameShape();
            out_w_over_height = truediv(inshape[1],inshape[0]);
            outshape[0]=(max_height);
            outshape[1]=max_height*out_w_over_height;
            if((math.floor(outshape[1])%2)==0):
                outshape[1]=math.floor(outshape[1])
            elif((math.ceil(outshape[1])%2)==0):
                outshape[1]=math.ceil(outshape[1])
            if((outshape[1]%2)!=0):
                outshape[1]=outshape[1]-1;
        sampling_rate = output_sampling_rate;
        if(not sampling_rate):
            sampling_rate=self.sampling_rate;

        tempfilepath = get_temp_file_path(final_file_path=output_path, temp_dir_path=self.getTempDir());
        self.openVideoWriter(output_file_path=tempfilepath, fps=output_sampling_rate);

        duration = self.getDuration();
        nsamples = sampling_rate*duration;
        old_frame_time = truediv(1.0,self.sampling_rate);
        frame_start_times = np.linspace(0,self.getDuration(),num=nsamples,endpoint=False);
        frame_index_floats = frame_start_times*self.sampling_rate;

        start_timer=time.time();
        last_timer=start_timer;
        fcounter=0;
        for nf in range(len(frame_index_floats)):
            fim = self.getImageFromFrame(frame_index_floats[nf]);
            if(reshape==True):
                fim.data = np.reshape(fim.data, (fim.data.shape[1],fim.data.shape[0],fim.data.shape[2]));
            if(outshape is not None):
                fim = fim.getScaled(shape=outshape);
            self.writeFrame(fim.data);

            fcounter+=1;
            if(not (fcounter%50)):
                if((time.time()-last_timer)>10):
                    last_timer=time.time();
                    print(("{}%% done after {} seconds...".format(100.0*truediv(fcounter,len(frame_index_floats)), last_timer-start_timer)));
        self.closeVideoWriter();
        rvid = Video.CreateFromVideoAndAudio(video_path=tempfilepath, audio_object=self.audio, output_path=output_path);
        os.remove(tempfilepath);
        if(input_sampling_rate_factor is not None):
            #return original_sampling_rate
            self.sampling_rate=original_sampling_rate;
            self.audio.sampling_rate = original_audio_sampling_rate;
        return rvid;

    def writeWarped(self, output_path, warp, output_sampling_rate=None, output_audio=None, bitrate=None, vbmark = True, max_time = None, **kwargs):
        sampling_rate = output_sampling_rate;
        if (not sampling_rate):
            sampling_rate = self.sampling_rate;

        duration = self.getDuration();
        old_frame_time = truediv(1.0, self.sampling_rate);

        target_start = warp.getTargetStart();
        target_end = warp.getTargetEnd();

        target_duration = target_end - target_start;

        if(max_time is not None and target_duration>max_time):
            target_duration = max_time;
            target_end = target_start+max_time;

        print((
            "target start: {}\ntarget end: {}\ntarget duration: {}".format(target_start, target_end, target_duration)));

        new_n_samples = target_duration * sampling_rate;
        target_start_times = np.linspace(target_start, target_end, num=new_n_samples, endpoint=False);
        unwarped_target_times = [];
        for st in target_start_times:
            unwarped_target_times.append(warp.warpTargetTime(st));
        frame_index_floats = np.true_divide(np.array(unwarped_target_times), old_frame_time);
        tempfilepath = get_temp_file_path(final_file_path=output_path, temp_dir_path=Video.VIDEO_TEMP_DIR);
        self.openVideoWriter(output_file_path=tempfilepath, fps=output_sampling_rate, **kwargs);
        start_timer = time.time();
        last_timer = start_timer;
        fcounter = 0;
        if(vbmark):
            vbmarker = self.getImageFromFrame(0)._vbmarker();

        for nf in range(len(frame_index_floats)):
            try:
                nwfr = self.getFrame(frame_index_floats[nf]);
                if(vbmark):
                    nwfrm = Image(data=nwfr);
                    nwfrm._splatAtPixCoord(**vbmarker);
                    nwfr = nwfrm._pixels_uint;
                self.writeFrame(nwfr);
            except ValueError:
                print(("VALUE ERROR ON WRITEFRAME {}".format(frame_index_floats[nf])));
                self.writeFrame(self.getFrame(math.floor(frame_index_floats[nf])));
            fcounter += 1;
            if (not (fcounter % 50)):
                if ((time.time() - last_timer) > 10):
                    last_timer = time.time();
                    print(("{}%% done after {} seconds...".format(100.0 * truediv(fcounter, len(frame_index_floats)),
                                                                 last_timer - start_timer)));
        self.closeVideoWriter();

        silent_warped_vid = Video(tempfilepath);
        if (not output_audio):
            output_audio = self.getAudio();

        cropped_output_audio = output_audio.AudioClip(start=target_start, end=target_end);

        if ((self.getInfo('max_height') is None) and (bitrate is None)):
            use_bitrate = "20000k";
            print(('Using bitrate of {}'.format(use_bitrate)));
            rvid = Video.CreateFromVideoAndAudioObjects(video=silent_warped_vid, audio=cropped_output_audio,
                                                        output_path=output_path, bitrate=use_bitrate);
        elif (bitrate is 'regular'):
            rvid = Video.CreateFromVideoAndAudioObjects(video=silent_warped_vid, audio=cropped_output_audio,
                                                        output_path=output_path);
        else:
            rvid = Video.CreateFromVideoAndAudioObjects(video=silent_warped_vid, audio=cropped_output_audio,
                                                        output_path=output_path, bitrate=bitrate);

        os.remove(tempfilepath);
        rvid.setInfo(label='warp_used', value=warp);
        return rvid;



    def getVersionLabel(self):
        return self.getInfo('version_label');

    def getWarpsDir(self):
        if(self.source is None):
            return self.getTempDir();
        version_label = self.getVersionLabel()
        return self.source.getWarpsDir(version_label=version_label);

    def getWithBeginningCroppedToAudio(self, target):
        if(isinstance(target, Audio)):
            B=target;
        else:
            B=target.getAudio();

        A = self.getAudio();

        AB = A.getShiftAmountTo(B);
        BA = B.getShiftAmountTo(A);

        if(AB<BA):
            return self.VideoClip(name=self.name+'clipped_to_'+B.name);
        else:
            return self.VideoClip(start=BA, end=None, name=self.name+'clipped_to_'+B.name);

    def getWarped(self, target,
                  source_events=None, target_events=None,
                  warp_type=None,
                  max_time = None,
                  source_time_range = None, target_time_range=None,
                  output_sampling_rate=None, output_path=None,
                  force_recompute=None, name_tag = None, vbmark = True, **kwargs):
        """

        :param target_audio:
        :param source_eventlist:
        :param target_eventlist:
        :param warp_type:
        :param source_time_range:
        :param target_time_range:
        :param output_sampling_rate:
        :param output_path:
        :param force_recompute:
        :param name_tag:
        :param kwargs:
        :return:
        """


        if(isinstance(target, Video)):
            target_audio = target.audio;
        else:
            target_audio = target;

        if(source_events is None):
            source_events = self.audio.getBeatEvents();
        if(target_events is None):
            target_events = target_audio.getBeatEvents();

        warp = Warp.FromEvents(source_events, target_events);

        if(warp_type is None):
            warp_type = 'quad';
        warp.setWarpFunc(warp_type, **kwargs);

        if(output_sampling_rate is None):
            output_sampling_rate = self.sampling_rate;

        if(hasattr(target, 'name')):
            warpname=self.name+'_TO_'+target.name;
        else:
            warpname = self.name+ '_TO_TARGETAUDIO' + '_' + str(target.getInfo('name'));

        if(name_tag is not None):
            warpname = warpname+name_tag;

        warpname = warpname+'.mp4';

        if(output_path is None):
            output_path = self.getWarpsDir()+os.sep+warpname;
            make_sure_dir_exists(output_path);

        if(not os.path.isfile(output_path) or force_recompute):
            rvid = self.writeWarped(output_path=output_path, warp=warp, output_sampling_rate=output_sampling_rate, output_audio=target_audio, vbmark = vbmark, max_time = max_time);
        else:
            rvid = Video(path=output_path, name=warpname);
        rvid.setFeature(name='warp_used', value=warp);
        return rvid;


    def getWithSoundsOnEvents(self, events, output_path=None, name_tag = None, mute_original=True,
                              event_sound=None, force_recompute=None, **kwargs):
        """

        :param events: Can be EventList, a list of events, or a list of numbers representing the times of the events.
        :param output_path: if None, will try to get from Video.source
        :param name_tag: what to label this output with in its name.
        :param mute_original: whether to leave the original sound from the video in the output.
        :param event_sound: Sound to play at each event.
        :param force_recompute:
        :param kwargs:
        :return:
        """

        assert(events is not None), 'must provide events or list of times to put sounds'

        output_name = self.getName()+'_eventsounds';
        if (name_tag):
            output_name = output_name + '_' + name_tag;
        if(output_path is None):
            if(self.source is not None):
                output_dir = self.source.getDirForVersion(version_label=self.getInfo('version_label'), version_group='sound_on_events');
            else:
                output_dir = self.getTempDir();
            output_path=os.path.join(output_dir, output_name+'.mp4');
            make_sure_dir_exists(output_dir);
        if(not os.path.isfile(output_path) or force_recompute):
            if(isinstance(events, EventList)):
                etimes = events.toStartTimes();
            elif(isinstance(events[0], Event)):
                etimes = Event.ToStartTimes(events);
            else:
                etimes = events;
            new_audio = self.getAudio().getWithSoundAdded(sound=event_sound, add_times=etimes,
                                                          mute_original=mute_original, **kwargs);
            audio_sig = new_audio.getSignal();
            reshapex = audio_sig.reshape(len(audio_sig), 1);
            reshapex = np.concatenate((reshapex, reshapex), axis=1);
            audio_clip = MPYAudioArrayClip(reshapex, fps=new_audio.sampling_rate);  # from a numeric arra
            video_clip = self.getMPYClip();
            video_clip = video_clip.set_audio(audio_clip);
            # video_clip.write_videofile(output_path, codec='libx264', write_logfile=False);
            MPYWriteVideoFile(video_clip, output_path, codec='libx264', write_logfile=False);

        rvid = Video(path=output_path, name=output_name);
        return rvid;


    @staticmethod
    def CreateFromVideoAndAudio(video_path=None, audio_path=None, video_object=None, audio_object=None,
                                output_path=None, clip_to_video_length=True, return_vid=True, codec='libx264',
                                bitrate=None, **kwargs):
        assert (not (video_path and video_object)), "provided both video path and object to CreateFromVideoAndAudio"
        assert (not (audio_path and audio_object)), "provided both audio path and object to CreateFromVideoAndAudio"
        assert (output_path), "Must provide output path for CreateFromVideoAndAudio"

        if (video_path):
            video_object = Video(video_path);
        if (audio_path):
            audio_object = Audio(path=audio_path);

        output_path = output_path.encode(sys.getfilesystemencoding()).strip();
        make_sure_dir_exists(output_path);

        # audio_sig = audio_object.getSignal();
        audio_sig = audio_object.getStereo();
        audio_sampling_rate = audio_object.getStereoSamplingRate();
        is_stereo = True;

        if (audio_sig is None):
            is_stereo = False;
            audio_sig = audio_object.getSignal();
            audio_sampling_rate = audio_object.sampling_rate;
            n_audio_samples_sig = len(audio_sig);
        else:
            n_audio_samples_sig = audio_sig.shape[1];

        print(("stereo is {}".format(is_stereo)));

        audio_duration = audio_object.getDuration();
        video_duration = video_object.getDuration();

        if (clip_to_video_length):
            n_audio_samples_in_vid = int(math.ceil(video_duration * audio_sampling_rate));

            if (n_audio_samples_in_vid < n_audio_samples_sig):
                if (is_stereo):
                    audio_sig = audio_sig[:, :int(n_audio_samples_in_vid)];
                else:
                    audio_sig = audio_sig[:int(n_audio_samples_in_vid)];
            else:
                if (n_audio_samples_in_vid > n_audio_samples_sig):
                    nreps = math.ceil(truediv(n_audio_samples_in_vid, n_audio_samples_sig));
                    if (is_stereo):
                        audio_sig = np.concatenate(
                            (audio_sig, np.zeros((2, n_audio_samples_in_vid - n_audio_samples_sig))),
                            axis=1);
                    else:
                        audio_sig = np.tile(audio_sig, (int(nreps)));
                        audio_sig = audio_sig[:int(n_audio_samples_in_vid)];

        if (is_stereo):
            # reshapex = np.reshape(audio_sig, (audio_sig.shape[1], audio_sig.shape[0]), order='F');
            reshapex = np.transpose(audio_sig);
            audio_clip = MPYAudioArrayClip(reshapex, fps=audio_sampling_rate);  # from a numeric arra
        else:
            reshapex = audio_sig.reshape(len(audio_sig), 1);
            reshapex = np.concatenate((reshapex, reshapex), axis=1);
            audio_clip = MPYAudioArrayClip(reshapex, fps=audio_sampling_rate);  # from a numeric arra

        video_clip = video_object.getMPYClip();
        video_clip = video_clip.set_audio(audio_clip);
        # video_clip.write_videofile(output_path,codec='libx264', write_logfile=False);
        if (bitrate is None):
            # video_clip.write_videofile(output_path, codec=codec, write_logfile=False);
            MPYWriteVideoFile(video_clip, output_path, codec=codec, write_logfile=False);
        else:
            MPYWriteVideoFile(video_clip, output_path, codec=codec, write_logfile=False, bitrate=bitrate);
            # video_clip.write_videofile(output_path, codec=codec, write_logfile=False, bitrate=bitrate);
        if (return_vid):
            return Video(output_path);
        else:
            return True;

    @staticmethod
    def CreateByStackingVideos(video_objects=None, video_paths=None, output_path=None, audio = None, concatdim = 0, force_recompute=True, **kwargs):
        assert output_path, "MUST PROVIDE OUTPUT PATH FOR VIDEO"
        if(not force_recompute):
            if(os.path.isfile(output_path)):
                return Video(path=output_path);
        matchdim = (concatdim+1)%2;
        vids=[];
        if(video_objects is not None):
            vids=video_objects;

        if(video_paths is not None):
            for vp in video_paths:
                vids.append(Video(path=video_paths));

        output_path=output_path.encode(sys.getfilesystemencoding()).strip();
        make_sure_dir_exists(output_path);
        basevid = vids[0];
        if(audio is None):
            audio = basevid.audio;
        sampling_rate = basevid.sampling_rate;
        tempfilepath = get_temp_file_path(final_file_path=output_path, temp_dir_path=vids[0].getTempDir());
        basevid.openVideoWriter(output_file_path=tempfilepath, fps=sampling_rate);
        duration = basevid.getDuration();
        nsamples = sampling_rate*duration;
        old_frame_time = truediv(1.0,sampling_rate);
        #frame_start_times = np.linspace(self.getStartTime(),self.getEndTime(),num=nsamples,endpoint=False);
        frame_start_times = np.linspace(0,duration,num=nsamples,endpoint=False);
        #frame_index_floats = frame_start_times/old_frame_time;
        frame_index_floats = frame_start_times*sampling_rate;
        for nf in range(len(frame_index_floats)):
            frameind = frame_index_floats[nf];
            newframe = basevid.getFrame(frameind);
            for vn in range(1,len(vids)):
                addpart = vids[vn].getFrame(frameind);
                partsize = np.asarray(addpart.shape)[:];
                cumulsize = np.asarray(newframe.shape)[:];
                if(partsize[matchdim]!=cumulsize[matchdim]):
                    sz = partsize[:];
                    sz[matchdim]=cumulsize[matchdim];
                    addpart = sp.misc.imresize(addpart, size=sz);
                newframe = np.concatenate((newframe, addpart), concatdim);
            basevid.writeFrame(newframe);
        basevid.closeVideoWriter();
        rvid = Video.CreateFromVideoAndAudio(video_path=tempfilepath, audio_object=audio, output_path=output_path, **kwargs);
        os.remove(tempfilepath);
        return rvid;

    @staticmethod
    def CreateFromVideoAndAudioPaths(video_path, audio_path, output_path, return_vid = True, **kwargs):
        return Video.CreateFromVideoAndAudio(video_path=video_path, audio_path=audio_path, output_path=output_path,return_vid=return_vid, **kwargs);

    @staticmethod
    def CreateFromVideoAndAudioObjects(video, audio, output_path, clip_to_video_length=True, return_vid = True, **kwargs):
        return Video.CreateFromVideoAndAudio(video_object=video, audio_object=audio, output_path=output_path, clip_to_video_length=clip_to_video_length, return_vid = return_vid, **kwargs);


    def _getDefaultPeakPickingTimeParams(self, **kwargs):
        single_frame = np.true_divide(1.0, self._getFrameRate());
        time_params = dict(
            pre_max_time=2.0 * single_frame,
            post_max_time=2.0 * single_frame,
            pre_avg_time=5.0 * single_frame,
            post_avg_time=5.0 * single_frame,
            wait_time=2.0 * single_frame,
            delta=0.015,
        )
        time_params.update(kwargs);
        return time_params;

#   ###################

    # FEATURE
    def getFrameIndexes(self, force_recompute=False):
        feature_name = 'frame_indexes';
        if ((not self.hasFeature(feature_name)) or force_recompute):
            params = dict();
            duration = self.getDuration();
            nsamples = self.sampling_rate * duration;
            frame_start_times = np.linspace(0, duration, num=nsamples, endpoint=False);
            value = frame_start_times * self.sampling_rate;
            self.setFeature(name=feature_name, value=value, params=params);
        return self.getFeature(feature_name);


    def getFeaturesList(self):
        return super(Video, self).getFeaturesList()+self.audio.getFeaturesList();

    def getVideoFeaturesList(self):
        return super(Video, self).getFeaturesList();

    def getFeatureFunctionsList(self):
        return super(Video, self).getFeatureFunctionsList()+self.audio.getFeatureFunctionsList();

    def getFeatureSourceType(self, name):
        """returns whether video or audio feature, instead of parent version which returns true or false"""
        vidhas = super(Video, self).hasFeature(name=name);
        if(vidhas):
            return 'video';
        audiohas = self.audio.hasFeature(name=name);
        if(audiohas):
            return 'audio';
        return None;

    def getFeature(self, name, force_recompute=False, **kwargs):
        """Returns None if feature is not already computed, and feature name is not implemented and registered with FEATURE_FUNCS"""
        params = kwargs;
        assert(not kwargs.get('params')), "STILL TRYING TO USE PARAMS INSTEAD OF KWARGS. FIX THIS";
        ftry = super(Video, self).getFeature(name=name, force_recompute=force_recompute, **kwargs);
        if(ftry is not None):
            return ftry;
        else:
            return self.audio.getFeature(name=name, force_recompute=force_recompute, **kwargs);


Video.FEATURE_FUNCS['frame_indexes']=Video.getFrameIndexes;
from . import Video_CV;
if(Video_CV.USING_OPENCV):
    Video.FEATURE_FUNCS.update(Video_CV.FEATURE_FUNCS);
    Video.USING_OPENCV = Video_CV.USING_OPENCV;

    Video.localRhythmicSaliencyFunction = Video_CV.localRhythmicSaliencyFunction
    Video.visualBeatFunction = Video_CV.visualBeatFunction

    Video.cvGetGrayFrame = Video_CV.cvGetGrayFrame;
    Video.getImageFromFrameGray = Video_CV.getImageFromFrameGray;
    Video.plotVisualBeats = Video_CV.plotVisualBeats;
    Video.loadFlowFeatures = Video_CV.loadFlowFeatures;

    Video.getVisualBeats = Video_CV.getVisualBeats
    Video.getLocalRhythmicSaliency = Video_CV.getLocalRhythmicSaliency;
    Video.getDirectogram = Video_CV.getDirectogram;
    Video.getDirectogramPowers = Video_CV.getDirectogramPowers;
    Video.getVisibleImpactEnvelope = Video_CV.getVisibleImpactEnvelope;
    Video.getVisibleImpactEnvelopePowers = Video_CV.getVisibleImpactEnvelopePowers;
    Video.getVisibleImpacts = Video_CV.getVisibleImpacts;
    Video.getVisualBeats = Video_CV.getVisualBeats;
    # Video.getBackwardVisualBeats = Video_CV.getBackwardVisualBeats;
    # Video.getForwardVisualBeats = Video_CV.getForwardVisualBeats;
    Video.getBackwardVisibleImpactEnvelope = Video_CV.getBackwardVisibleImpactEnvelope;
    Video.getBothWayVisibleImpactEnvelope = Video_CV.getBothWayVisibleImpactEnvelope;
    Video.getForwardVisibleImpactEnvelope = Video_CV.getForwardVisibleImpactEnvelope;
    Video.getDirectionalFlux = Video_CV.getDirectionalFlux;
    Video.getVisualTempogram = Video_CV.getVisualTempogram;
    Video.getCutEvents = Video_CV.getCutEvents;
    Video.computeImpactEnvelope = Video_CV.computeImpactEnvelope;
    Video.computeDirectogramPowers = Video_CV.computeDirectogramPowers;
    Video.visualBeatsFromEvents = Video_CV.visualBeatsFromEvents;
    Video.plotVisualBeats = Video_CV.plotVisualBeats;
    Video.plotImpactEnvelope = Video_CV.plotImpactEnvelope;
    Video.plotVisibleImpacts = Video_CV.plotVisibleImpacts;
    Video.getVisualBeatSequences = Video_CV.getVisualBeatSequences;
    Video.printVisualBeatSequences = Video_CV.printVisualBeatSequences;
    Video.getVisualBeatTimes = Video_CV.getVisualBeatTimes;
    Video.findAccidentalDanceSequences = Video_CV.findAccidentalDanceSequences;
    Video.plotEvents = Video_CV.plotEvents;
    # Video. = Video_CV.
    # Video. = Video_CV.
    # Video. = Video_CV.
    # Video. = Video_CV.
    # Video. = Video_CV.
    # Video. = Video_CV.

else:
    AWARN("Was not able to add functions that use OpenCV! Check OpenCV instalation and try again?");

from .vbgui import BeatGUI
if(BeatGUI.VIEWER_INSTALLED):
    def getEvents(self, **kwargs):
        time_params = self._getDefaultPeakPickingTimeParams();
        time_params.update(kwargs)
        vbeats = self.getFeature('visual_beats', force_recompute=True, **time_params);
        return vbeats;
    def getEventList(self, **kwargs):
        events = self.getEvents(**kwargs);
        return EventList(events=events);
    def runBeatGUIOnAudio(self):
        audio = self.getAudio();
        self.gui.run(local_saliency=audio.getLocalRhythmicSaliency(), frame_rate = audio._getFrameRate(), eventlist = audio.getEventList());


    def runGUI(self, local_saliency=None, frame_rate = None, eventlist = 'default', frame_offset=None):
        self.gui.run(local_saliency=local_saliency, frame_rate=frame_rate, eventlist=eventlist, frame_offset=frame_offset);

    Video._getGui = BeatGUI.media_GUI_func;
    Video.runGUI = runGUI;
    Video.getEvents = getEvents;
    Video.getEventList = getEventList;
    Video.runBeatGUIOnAudio = runBeatGUIOnAudio;

else:
    AWARN("BeatGUI not installed");