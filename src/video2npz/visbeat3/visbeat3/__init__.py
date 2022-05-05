
from .VisBeatImports import *
from .Video import *
from .VideoClip import *
from .VideoSource import *
from .VisBeatExampleVideo import VisBeatExampleVideo
import re
import os;
import shutil


from .SourceLocationParser import ParseSourseLocation

VISBEAT_ASSETS_DIR = './VisBeatAssets/';

from . import fileui
fileui.INITIAL_DIR = VISBEAT_ASSETS_DIR;


def SetAssetsDir(assets_dir):
    global VISBEAT_ASSETS_DIR;
    VISBEAT_ASSETS_DIR = assets_dir;
    make_sure_dir_exists(assets_dir);
    AINFORM("VISBEAT_ASSETS_DIR set to {}".format(VISBEAT_ASSETS_DIR));
    make_sure_dir_exists(GetVideoSourcesDir());
    temp_dir = os.path.join(VISBEAT_ASSETS_DIR, 'TEMP_FILES'+os.sep);
    make_sure_dir_exists(temp_dir);
    Video.VIDEO_TEMP_DIR = temp_dir;
    fileui.INITIAL_DIR = VISBEAT_ASSETS_DIR;

def GetAssetsDir():
    return VISBEAT_ASSETS_DIR;

def GetVideoSourcesDir():
    video_sources_dir = os.path.join(GetAssetsDir(), 'VideoSources'+os.sep);
    make_sure_dir_exists(video_sources_dir);
    return video_sources_dir;

def PullVideo(name=None, source_location=None, max_height=240, **kwargs):
    if(isinstance(name, VisBeatExampleVideo)):
        assert(source_location is None), 'Provided VisBeatExampleVideo and source location? What are you trying to do?';
        source_location = name.url;
        vname = name.name;
    elif(name is None):
        assert(source_location is not None), "Must provide an argument to pullvideo";
        sloc = ParseSourseLocation(source_location);
        vname =sloc.code;
    else:
        vname = name;

    vs = GetVideoSource(vname);
    if(vs and vs.source_location==source_location):
        v = vs.getVersion(max_height=max_height);
        v.load(features_to_load = 'all');
        return v;
    
    print("destination:", GetVideoSourcesDir(), "name:", vname, "source_location:", source_location)
    vs = VideoSource.NewVideoSource(destination=GetVideoSourcesDir(), name=vname, source_location=source_location, max_height=max_height, **kwargs);
    v = vs.getVersion(max_height=max_height);
    return v;

def ClipVideo(video, time_range, max_height=240):
    video_fullres = video.source.getVersion();
    vclip = video_fullres.VideoClip(start=time_range[0], end=time_range[1]);
    vcdir = video.source.getDirForVersion(version_label='{}_{}'.format(str(time_range[0]), str(time_range[1])), version_group='Clips');
    make_sure_dir_exists(vcdir);
    vcname = video.getName() + 'clip_{}_{}'.format(str(time_range[0]), str(time_range[1]));
    vcpath = os.path.join(vcdir, vcname+'.mp4');
    vclip.write(output_path=vcpath);
    vs = VideoSource.NewVideoSource(destination=GetVideoSourcesDir(), name=vcname, source_location=vcpath);
    return vs.getVersion(max_height=max_height);


def GetVideoSource(name):
    vname = name;
    if (isinstance(name, VisBeatExampleVideo)):
        vname = name.name;
    path = os.path.join(GetVideoSourcesDir(), vname) + os.sep;
    if (os.path.isdir(path)):
        return VideoSource(path=path);

def LoadVideo(name, max_height=240):
    vname = name;
    if (isinstance(name, VisBeatExampleVideo)):
        vname = name.name;

    path = os.path.join(GetVideoSourcesDir(), vname)+os.sep;
    if(os.path.isdir(path)):
        vs = VideoSource(path=path);
        v = vs.getVersion(max_height=max_height);
        v.load(features_to_load = 'all');
        return v;
    else:
        return None;

def Dancefer(source_video, target,
             synch_video_beat=0, synch_audio_beat=0,
             beat_offset = 0, leadin = None, nbeats=None,
             source_harmonic = None, target_harmonic = None, source_harmonic_offset=None, target_harmonic_offset=None,
             force_recompute=None, warp_type = 'quad',
             name_tag=None, name_tag_prefix=None, output_path = None,
             **kwargs):
    """

    :param source_video: video to warp
    :param target: music to warp to
    :param synch_video_beat: integer specifying a beat (as in the nth beat) from the video to synchronize with synch_audio_beat
    :param synch_audio_beat: integer specifying a beat (as in the nth beat) from the video to synchronize with synch_video_beat
    :param beat_offset: Lets you offset which beats you want to render. This is mostly for testing different parts of an output.
    :param leadin: how many beats before the synch beats to render
    :param nbeats: lets you restrict output to rendering n beats
    :param source_harmonic: can be None, 'half', or 'double'. 'half' will use every other beat, which you can offset with source_harmonic_offset. 'double' will add an additional beat between every consecutive beat. update - added 'third' for waltzes.
    :param target_harmonic: can be None, 'half', or 'double'. 'half' will use every other beat, which you can offset with source_harmonic_offset. 'double' will add an additional beat between every consecutive beat. update - added 'third' for waltzes.
    :param source_harmonic_offset: optional offset for harmonic
    :param target_harmonic_offset: optional offset for harmonic
    :param force_recompute:
    :param warp_type:
    :param name_tag:
    :param name_tag_prefix:
    :param output_path:
    :param kwargs:
    :return:
    """


    if((output_path is not None) and (not force_recompute)):
        if(os.path.exists(output_path)):
            return Video(output_path);

    if(isinstance(target, Video)):
        target_audio = target.getAudio();
    else:
        target_audio = target;


    synchaudio = synch_audio_beat;
    synchvideo = synch_video_beat;
    lead_in = leadin;
    if(lead_in is None):
        lead_in = min(synchaudio, synchvideo);
    elif(isinstance(lead_in, str) and lead_in[0]=='<'):
        # lead_in = min(synchaudio, synchvideo, int(lead_in));
        lead_in = min(synchaudio, int(lead_in));

    start_audio_beat = synchaudio-lead_in;
    start_video_beat = synchvideo-lead_in;

    if(beat_offset and beat_offset>0):
        start_audio_beat = start_audio_beat+beat_offset;
        start_video_beat = start_video_beat+beat_offset;

    print(("Warping {} to {}".format(source_video.getName(), target_audio.getName())));
    bitrate = None;
    vbeats = source_video.audio.getBeatEvents()
    tbeats = target_audio.getBeatEvents()

    if(start_video_beat < 0):
        if(synchvideo == 0):
            vbeats = [vbeats[0].clone()]+vbeats;
            vbeats[0].start = vbeats[0].start-(vbeats[2].start-vbeats[1].start);
        vbadd = Event.SubdivideIntervals(vbeats[:2], -start_video_beat);
        vbeats = vbadd+vbeats[2:];
        start_video_beat = 0;


    vbeats = vbeats[start_video_beat:];
    tbeats = tbeats[start_audio_beat:];

    if(source_harmonic=='half'):
        vbeats = Event.Half(vbeats, source_harmonic_offset);
    elif (source_harmonic == 'third'):
        vbeats = Event.Third(vbeats, source_harmonic_offset);
    elif(source_harmonic == 'double'):
        vbeats = Event.Double(vbeats);

    if (target_harmonic == 'half'):
        tbeats = Event.Half(tbeats, target_harmonic_offset);
    elif (target_harmonic == 'third'):
        tbeats = Event.Third(tbeats, target_harmonic_offset);
    elif (target_harmonic == 'double'):
        tbeats = Event.Double(tbeats);


    if(nbeats):
        print(("Rendering {} beats of result".format(nbeats)))
        if(len(vbeats)>nbeats):
            vbeats = vbeats[:nbeats];
            print((len(vbeats)))
        if(len(tbeats)>nbeats):
            tbeats = tbeats[:nbeats];
            print((len(tbeats)))
    else:
        if(vbeats[-1].start<source_video.getDuration()):
            print(tbeats)
            print(("length of tbeats is: {}".format(len(tbeats))));
            print(("start_video_beat: {}, start_audio_beat: {}".format(start_video_beat, start_audio_beat)))
            newbeat = vbeats[-1].clone();
            deltatime = source_video.getDuration()-newbeat.start;
            newbeat.start = source_video.getDuration();
            target_newbeat = tbeats[-1].clone();
            target_newbeat.start = min(target_newbeat.start+deltatime, target_audio.getDuration());
            tbeats.append(target_newbeat);

    if(warp_type is 'weight'):
        vbeats = source_video.visualBeatsFromEvents(vbeats);

    if(name_tag is None):
        name_tag = warp_type+'_sab_'+str(start_audio_beat)+'_svb_'+str(start_video_beat);
    if(name_tag_prefix is not None):
        name_tag = name_tag+name_tag_prefix;

    warp_args = dict(target=target_audio,
                     source_events=vbeats,
                     target_events = tbeats,
                     warp_type=warp_type,
                     force_recompute=force_recompute,
                     name_tag = name_tag)
    if(bitrate):
        warp_args.update(dict(bitrate=bitrate));

    warp_args.update(kwargs);
    warped_result = source_video.getWarped(**warp_args);

    if(output_path):
        final_output_path = output_path;
        if(os.path.isfile(final_output_path)):
            output_filename = os.path.basename(output_path);
            name_parts = os.path.splitext(output_filename);
            output_filename_base = name_parts[0];
            output_directory_path = os.path.dirname(output_path);
            if (output_directory_path == ''):
                output_directory_path = '.'
            output_ext = name_parts[1];
            ntry = 1;
            tryname = output_filename_base+ '_' + str(ntry);
            while (os.path.isfile(os.path.join(output_directory_path, tryname+output_ext)) and ntry<100):
                ntry = ntry+1;
                tryname = output_filename_base + '_' + str(ntry);

            final_output_path = os.path.join(output_directory_path, tryname + output_ext);
        shutil.copy2(src=warped_result.getPath(), dst=final_output_path);
        n_frames_total = warped_result.num_frames_total;
        warp_used = warped_result.getInfo('warp_used');
        warped_result_final = Video(path = final_output_path, num_frames_total=n_frames_total);
        warped_result_final.setInfo(label='warp_used', value=warp_used);
        os.remove(warped_result.getPath())
        warped_result = warped_result_final;
    return warped_result;

def get_temp_file_path(final_file_path="TEMP", temp_dir_path = None):
    pparts = os.path.split(final_file_path);
    destfolder = pparts[0]+os.sep;
    tempdir = temp_dir_path;
    if(tempdir is None):
        tempdir='.';
    destfolder=pathstring(tempdir+os.sep);
    tempname = 'TEMP_'+pparts[1];
    temptry = 0;
    while(os.path.isfile(destfolder+tempname)):
        temptry=temptry+1;
        tempname = 'TEMP{}_'.format(temptry)+pparts[1];
    return pathstring(destfolder+tempname);


def AutoDancefer(source, target, output_path = None, synch_video_beat = 0, synch_audio_beat = 0, beat_offset = 0, **kwargs):
    sourcev = PullVideo(source_location=source);
    targetv = PullVideo(source_location=target);

    result =  Dancefer(source_video=sourcev, target=targetv, output_path=output_path, force_recompute = True, synch_audio_beat=synch_audio_beat, synch_video_beat=synch_video_beat, beat_offset=beat_offset,**kwargs)
    AINFORM("\n\n\nResult saved to {}\n\n\n".format(result.getPath()));
    return result;

#########


def Dancify(source_video, target,
            source_beats=None, target_beats=None,
            synch_video_beat=0, synch_audio_beat=0,
            beat_offset = 0, leadin = None, nbeats=None,
            unfold_to_n=None,
            source_harmonic = None, source_harmonic_offset=None,
            target_harmonic = None, target_harmonic_offset=None,
            force_recompute=None, warp_type = 'quad',
            name_tag=None, name_tag_prefix=None, output_path = None,
            momentum = 0.1,
            **kwargs):
    """

    :param source_video:
    :param target:
    :param source_beats:
    :param target_beats:
    :param synch_video_beat:
    :param synch_audio_beat:
    :param beat_offset:
    :param leadin:
    :param nbeats:
    :param unfold_to_n:
    :param source_harmonic:
    :param target_harmonic:
    :param source_harmonic_offset:
    :param target_harmonic_offset:
    :param force_recompute:
    :param warp_type:
    :param name_tag:
    :param name_tag_prefix:
    :param output_path:
    :param momentum:
    :param kwargs:
    :return:
    """


    if((output_path is not None) and (not force_recompute)):
        if(os.path.exists(output_path)):
            return Video(output_path);

    if(isinstance(target, Video)):
        target_audio = target.getAudio();
    else:
        target_audio = target;


    synchaudio = synch_audio_beat;
    synchvideo = synch_video_beat;
    lead_in = leadin;
    if(lead_in is None):
        lead_in = min(synchaudio, synchvideo);
    elif(isinstance(lead_in, str) and lead_in[0]=='<'):
        # lead_in = min(synchaudio, synchvideo, int(lead_in));
        lead_in = min(synchaudio, int(lead_in));

    start_audio_beat = synchaudio-lead_in;
    start_video_beat = synchvideo-lead_in;

    if(beat_offset and beat_offset>0):
        start_audio_beat = start_audio_beat+beat_offset;
        start_video_beat = start_video_beat+beat_offset;

    print(("Warping {} to {}".format(source_video.getName(), target_audio.getName())));
    bitrate = None;
    vbeats = source_beats;
    if(source_beats is None):
        vbeats = source_video.getVisualBeats();


    tbeats = target_beats;
    if(target_beats is None):
        tbeats = target_audio.getBeatEvents();

    if(start_video_beat < 0):
        if(synchvideo == 0):
            vbeats = [vbeats[0].clone()]+vbeats;
            vbeats[0].start = vbeats[0].start-(vbeats[2].start-vbeats[1].start);
        vbadd = Event.SubdivideIntervals(vbeats[:2], -start_video_beat);
        vbeats = vbadd+vbeats[2:];
        start_video_beat = 0;


    vbeats = vbeats[start_video_beat:];
    tbeats = tbeats[start_audio_beat:];

    if (source_harmonic == 'half'):
        vbeats = Event.Half(vbeats, source_harmonic_offset);
    elif (source_harmonic == 'third'):
        vbeats = Event.Third(vbeats, source_harmonic_offset);
    elif (source_harmonic == 'double'):
        vbeats = Event.Double(vbeats);

    if (target_harmonic == 'half'):
        tbeats = Event.Half(tbeats, target_harmonic_offset);
    elif (target_harmonic == 'third'):
        tbeats = Event.Third(tbeats, target_harmonic_offset);
    elif (target_harmonic == 'double'):
        tbeats = Event.Double(tbeats);


    if(nbeats):
        print(("Rendering {} beats of result".format(nbeats)))
        if(len(vbeats)>nbeats):
            vbeats = vbeats[:nbeats];
            print((len(vbeats)))


    if(unfold_to_n):
        vbeats = Event.UnfoldToN(vbeats, unfold_to_n, momentum=momentum);

    if (len(tbeats) > len(vbeats)):
        tbeats = tbeats[:len(vbeats)];

    if(warp_type is 'weight'):
        vbeats = source_video.visualBeatsFromEvents(vbeats);

    if(name_tag is None):
        name_tag = warp_type+'_sab_'+str(start_audio_beat)+'_svb_'+str(start_video_beat);
    if(name_tag_prefix is not None):
        name_tag = name_tag+name_tag_prefix;

    warp_args = dict(target=target_audio,
                     source_events=vbeats,
                     target_events = tbeats,
                     warp_type=warp_type,
                     force_recompute=force_recompute,
                     name_tag = name_tag)
    if(bitrate):
        warp_args.update(dict(bitrate=bitrate));

    warp_args.update(kwargs);
    warped_result = source_video.getWarped(**warp_args);

    if(output_path):
        final_output_path = output_path;
        if(os.path.isfile(final_output_path)):
            output_filename = os.path.basename(output_path);
            name_parts = os.path.splitext(output_filename);
            output_filename_base = name_parts[0];
            output_directory_path = os.path.dirname(output_path);
            if (output_directory_path == ''):
                output_directory_path = '.'
            output_ext = name_parts[1];
            ntry = 1;
            tryname = output_filename_base+ '_' + str(ntry);
            while (os.path.isfile(os.path.join(output_directory_path, tryname+output_ext)) and ntry<100):
                ntry = ntry+1;
                tryname = output_filename_base + '_' + str(ntry);

            final_output_path = os.path.join(output_directory_path, tryname + output_ext);
        shutil.copy2(src=warped_result.getPath(), dst=final_output_path);
        n_frames_total = warped_result.num_frames_total;
        warp_used = warped_result.getInfo('warp_used');
        warped_result_final = Video(path = final_output_path, num_frames_total=n_frames_total);
        warped_result_final.setInfo(label='warp_used', value=warp_used);
        os.remove(warped_result.getPath())
        warped_result = warped_result_final;
    return warped_result;



    def getVBSegments(self,source_video,
                      source_beats = None,
                      search_tempo=None,
                      search_window=None,
                      max_height=None,
                      beat_limit=None,
                      n_return=None,
                      unary_weight=None,
                      binary_weight=None,
                      break_on_cuts=None,
                      peak_vars=None):

        source = source_video;
        if(source_beats is None):
            if (peak_vars is not None):
                vbeats = source.getFeature('simple_visual_beats', **peak_vars);
            else:
                vbeats = source.getFeature('simple_visual_beats');
        else:
            vbeats = source_beats;

        #

        if (search_tempo is not None):
            tempo = search_tempo;
            beat_time = np.true_divide(60.0, tempo);
            clips = VisualBeat.PullOptimalPaths_Basic(vbeats, target_period=beat_time, unary_weight=unary_weight,
                                                      binary_weight=binary_weight, break_on_cuts=break_on_cuts,
                                                      window_size=search_window);
        else:
            clips = VisualBeat.PullOptimalPaths_Autocor(vbeats, unary_weight=unary_weight, binary_weight=binary_weight,
                                                        break_on_cuts=break_on_cuts, window_size=search_window);

        if (beat_limit is None):
            beat_limit = 2;

        print(("There were {} candidates".format(len(vbeats))));
        nclips = 0;
        segments = [];

        for S in clips:
            if (len(S) > beat_limit):
                nclips = nclips + 1;
                segments.append(S);

        if (n_return is not None):
            segments.sort(key=len, reverse=True);
            segments = segments[:n_return];
        return segments;