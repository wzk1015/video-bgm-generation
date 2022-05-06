from .VisBeatImports import *
from .Video import *
from .AFileManager import AFileManager

try:
    import youtube_dl
except ImportError:
    AWARN('You need to install youtube-dl to use parts of VideoSource class that pull from YouTube. Try running:\npip install youtube-dl')



def safe_file_name(input_string):
    return ''.join([i if ord(i) < 128 else '_' for i in input_string]);

class VideoSource(AFileManager):
    """Video (class): A video, and a bunch of convenience functions to go with it.
        Attributes:
            name: The name of the video
            sampling_rate: The framerate of the video
            audio: an Audio object, the audio for the video
    """

    VIDEO_TEMP_DIR = './'

    def getJSONName(self):
        return self.AOBJECT_TYPE()+".json";

    def AOBJECT_TYPE(self):
        return 'VideoSource';

    def __init__(self, path, name=None, source_location=None, VideoClass = None, **kwargs):
        """
        :param path: either the path to a json, or the path to a directory containing 'VideoSource.json',
            or the path to a directory where said json should be created.
        :param name: name for video
        :param source_location: can be a path to a video file, or a youtube source from which to pull a video file
        :param VideoClass: visbeat.Video by default, but can be changed if you are using a custom subclass.
               Must be a subclass of visbeat.Video, and must be constructable with VideoClass(path)
        """
        AFileManager.__init__(self, path=path);

        if(VideoClass is not None):
            self.VideoClass = VideoClass;

        self.setSource(source_location=source_location, assert_valid=None, **kwargs);

        # if (self.name is None):
            # self.name = os.path.splitext(os.path.basename(path))[0]


    @staticmethod
    def NewVideoSource(destination, name, source_location=None, VideoClass = None, **kwargs):
        vsdir = os.path.join(destination, name+os.sep);
        make_sure_dir_exists(vsdir);
        print(("Video source at {}".format(vsdir)));
        return VideoSource(path=vsdir, name=name, source_location=source_location, **kwargs);

    def initializeBlank(self):
        AFileManager.initializeBlank(self);
        self.source_location = None;
        self.name = None;
        self.video_file_name = None;
        self.title_safe = None;
        self.youtube_info_dict = None;
        self.versions_info = {};

        #
        self.VideoClass = Video;

    def toDictionary(self):
        d = AFileManager.toDictionary(self);
        d['source_location']=self.source_location;
        if(self.name):
            d['name']=self.name;
        if(self.video_file_name):
            d['video_file_name']=self.video_file_name;
        if(self.title_safe):
            d['title_safe']=self.title_safe;
        if(self.youtube_info_dict):
            d['youtube_info_dict']=self.youtube_info_dict;
        if(self.versions_info):
            d['versions_info']=self.versions_info;
        return d;


    def initFromDictionary(self, d):
        AFileManager.initFromDictionary(self, d);
        self.source_location = d['source_location'];
        self.name = d.get('name');
        self.video_file_name = d.get('video_file_name');
        self.title_safe = d.get('title_safe');
        self.youtube_info_dict = d.get('youtube_info_dict');
        self.versions_info=d.get('versions_info');
        if(self.versions_info is None):
            self.versions_info = {};
        #do class specific inits with d;


    def initWithPath(self, path=None, clear_temp=None):
        AFileManager.initWithPath(self, path=path, clear_temp=clear_temp);
        self.setDir('versions', pathstring(self.getDirectoryPath() + os.sep + "Versions" + os.sep));
        self.setDir('warps', pathstring(self.getDir('versions')+ os.sep + "Warps" + os.sep));
        # self.setDir('midi', pathstring(self.getDirectoryPath() + os.sep + "MIDI" + os.sep));
        # self.setDir('music', pathstring(self.getDirectoryPath() + os.sep + "Music" + os.sep));
        self.setFeaturesDir();

    def setFeaturesDir(self, features_dir=None):
        if (features_dir is None):
            self.setDir('features', pathstring(self.getDir('data') + os.sep + "Features" + os.sep))
        else:
            self.setDir('features', features_dir);



    def getName(self):
        """
        Gets name if set. If not set, returns file name without extension.
        :return:
        """
        if(self.name is not None):
            return self.name;
        elif(self.video_file_name is not None):
            return os.path.splitext(self.video_file_name)[0];
        else:
            return None;


###################


    @staticmethod
    def _versionLabelString(version_label=None):
        if (version_label and version_label != 'Full'):
            return str(version_label);
        else:
            return 'Full';

    @staticmethod
    def _versionGroupString(version_group=None):
        if (version_group is None):
            version_group = 'Original';
        return version_group;

    @staticmethod
    def _getVersionLabelDirName(version_label=None):
        if(isinstance(version_label, int)):
            return ("maxheight_{}".format(version_label));
        else:
            return "Full";


    def getVersionPath(self, version_label=None, version_group=None):
        """
        Get path to version of video (path to video file). Return None if version has not been added.
        :param version_label:
        :return:
        """
        return self.getVersionInfo(version_label=version_label, version_group=version_group, info_label='path');

    def getDirForVersion(self, version_label=None, version_group=None):
        vdir = self.getDir('versions') + os.sep + self._versionGroupString(version_group) + os.sep + self._getVersionLabelDirName(version_label)+os.sep;
        return vdir;


    def getVersionInfo(self, version_label, version_group=None, info_label=None):
        """
        Get info about a version of the video
        :param version_label:
        :param info_label:
        :return:
        """
        assert(info_label is not None), "should provide info_label to getVersionInfo()"
        d = self.getVersionDictionary(version_label=version_label, version_group=version_group);
        if(d is not None):
            return d.get(info_label);
        else:
            return None;

    def getVersionDictionary(self, version_label=None, version_group=None):
        if(version_group is None):
            version_group='Original';
        vis_d=self.versions_info.get(version_group);
        if(vis_d is not None):
            return vis_d.get(VideoSource._versionLabelString(version_label));
        else:
            return None;

    def setVersionDictionary(self, version_label=None, version_group=None, d=None):
        if(version_group is None):
            version_group='Original';
        vis_d=self.versions_info.get(version_group);
        if(vis_d is None):
            self.versions_info[version_group]={};
            vis_d=self.versions_info.get(version_group);
        vis_d[VideoSource._versionLabelString(version_label)]=d;
        return;

    def setVersionInfo(self, version_label=None, version_group=None, video_path=None, **kwargs):
        version_dict = self.getVersionDictionary(version_label=version_label, version_group=version_group);
        if(version_dict is None):
            self.setVersionDictionary(version_label=version_label, version_group=version_group, d={});
            version_dict = self.getVersionDictionary(version_label=version_label, version_group=version_group);
        if(video_path is not None):
            version_dict.update({'path':str(pathstring(video_path))});
        if(kwargs is not None):
            version_dict.update(kwargs);


####################

    def hardSave(self):
        if (os.path.isfile(self.getJSONPath())):
            os.rename(self.getJSONPath(), self.getDir('backup') + os.sep + self.AOBJECT_TYPE() + ".json");
        self.writeToJSON(self.getJSONPath());

    def save(self):
        if (self.getInfo('block_writing_to_json')):
            return;
        self.hardSave();

####################


    # def removeAllVersionFiles(self, asset_manager=None):
    #     AWARN("This should wipe the directory, ya?")
    #     # AWARN("Remove all version files does not remove features and vis images! This still needs to be implemented!")
    #     # for vtype in self.versions_info:
    #     #     for v in self.versions_info.get(vtype):
    #     #         # print(self.versions_info.get(vtype).get(v));
    #     #         vpath = self.versions_info.get(vtype).get(v).get('path');
    #     #         assert(vpath), "version {} did not have path".format(vtype)
    #     #         print(vpath)
    #     #         if(os.path.isfile(vpath)):
    #     #             print("REMOVING {}".format(vpath))
    #     #             os.remove(vpath);
    #     #         if(asset_manager is not None and vtype=='Original'):
    #     #             if(v=='Full'):
    #     #                 rs=None;
    #     #             else:
    #     #                 rs = int(v);
    #     #             asset_manager.removeFeaturesForLinkRes(link=self, max_height=rs);
    #     # return True;

    # def removeFeaturesForRes(self, max_height=None):
    #     v = self.getVersionVideo(name=link.name, max_height = max_height);
    #     v.clearFeatureFiles(features_to_clear=['all', 'each']);
    #
    #
    # def removeFeatureFilesForVideo(self, video, features_to_remove=None):
    #     if(features_to_remove is None):
    #         return;
    #     if(not isinstance(features_to_remove, list)):
    #         features_to_remove=[features_to_remove];
    #     for f in features_to_remove:
    #         self.removeFeatureFileForVideo(video=video, feature_name=f);
    #
    #
    # def removeFeatureFileForVideo(self, video, feature_name):
    #     AWARN("still need to implement removeFeatureFileForVideo. Should just empty corresponding directory. Also implement 'each' that removes all.")
    #     # if(feature_name=='each'):
    #     #     return self.removeFeatureFilesForVideo(video=video, features_to_remove=video.getFeatureFunctionsList());
    #     # max_height = video.getInfo(label='max_height');
    #     # link = self.hasLinkWithName(name=video.name);
    #     # source_type=video.getFeatureSohahurceType(feature_name);
    #     # ipath = self.getFeaturePathForLink(feature_name=feature_name, link=link, max_height=max_height, source_type=source_type);
    #     # if(os.path.isfile(ipath)):
    #     #     print("REMOVING {}".format(ipath));
    #     #     os.remove(ipath);
    #
    # def removeFeatureFiles(self, video, feature_name):
    #     if(feature_name=='each'):
    #         AWARN("IMPLEMENT DELETE FEATURE DIR AND CREATE CLEAN");
    #     version_label = video.getInfo(label='version_label');
    #     AWARN("IMPLEMENT DELETE {} FOR VERSION_LABEL {}".format(feature_name, version_label));
    #     # ipath = self.getFeaturePathForLink(feature_name=feature_name, link=link, max_height=max_height, source_type=source_type);
    #     # if(os.path.isfile(ipath)):
    #     #     print("REMOVING {}".format(ipath));
    #     #     os.remove(ipath);
    #
    # def removeVersion(self, version_label=None, remove_files=True):
    #     AWARN("Remove Resolution does not remove features! This still needs to be implemented!")
    #     for vtype in self.versions_info:
    #         v = self.versions_info.get(vtype).get(VideoSource._versionLabelString(version_label));
    #         if(v is not None):
    #             vpath = v.get('path');
    #             if(os.path.isfile(vpath)):
    #                 print("REMOVING {}".format(vpath))
    #                 os.remove(vpath);
    #             self.versions_info.get(vtype).pop(VideoSource._versionLabelString(version_label), None);
    #     return True;
    #
    # def copyResolutionTo(self, version_label=None, output_dir=None):
    #     assert(output_dir)
    #     for vtype in self.versions_info:
    #         v = self.versions_info.get(vtype).get(VideoSource._versionLabelString(version_label));
    #         if(v is not None):
    #             vpath = v.get('path');
    #             if(os.path.isfile(vpath)):
    #                 opath = os.path.join(output_dir,(vtype+'_'+VideoSource._versionLabelString(version_label)+self.video_file_name));
    #                 print("COPYING {} to {}".format(vpath,opath));
    #                 shutil.copy2(vpath, opath);
    #     return True;
    #
    # def copyAssetsTo(self, output_dir=None, asset_manager=None):
    #     for vtype in self.versions_info:
    #         output_type_dir=pathstring(output_dir+os.sep+vtype+os.sep);
    #         make_sure_dir_exists(output_type_dir);
    #         for v in self.versions_info.get(vtype):
    #             output_a_dir = pathstring(output_type_dir+os.sep+v+os.sep);
    #             make_sure_dir_exists(output_a_dir);
    #             # print(self.versions_info.get(vtype).get(v));
    #             vpath = self.versions_info.get(vtype).get(v).get('path');
    #             if(vpath and os.path.isfile(vpath)):
    #                 print("copying {} to {}".format(vpath, output_a_dir));
    #                 shutil.copy2(vpath, output_a_dir);
    #                 if(asset_manager is not None and vtype=='Original'):
    #                     if(v=='Full'):
    #                         rs=None;
    #                     else:
    #                         rs = int(v);
    #                     asset_manager.saveFeaturesForVersion(version_label=rs, output_dir = output_a_dir);

    def getWarpsDir(self, version_label=None):
        warpdir = self.getDir('warps');
        resdir = self._getVersionLabelDirName(version_label=version_label);
        rdir = pathstring(warpdir + os.sep + resdir + os.sep);
        make_sure_path_exists(rdir);
        return rdir;

    # ############################# FROM ASSETMANAGER #######################

    def getVersion(self, max_height=None, get_if_missing=True, load_features=True, **kwargs):
        """
        Gets a version of the video with maximum height max_height. This function contains logic to decide whether/when
            to pull the video. Use pullVersion to force pulling that overwrite's existing assets.
        :param max_height:
        :param get_if_missing:
        :return:
        """
        vpath = self.getVersionPath(version_label=max_height);
        if(vpath and os.path.isfile(vpath)):
            num_frames_total = self.getVersionInfo(version_label=max_height, info_label='num_frames_total');
            if(num_frames_total):
                v = self.RegisteredVideo(path=vpath, version_label = max_height, num_frames_total=num_frames_total, load_features=True);
            else:
                v = self.RegisteredVideo(path=vpath, version_label = max_height, load_features=True);
                self.setVersionInfo(version_label=max_height, num_frames_total=v.num_frames_total);
                self.save();
            return v;
        else:

            if(get_if_missing):
                vpath = self.pullVersion(max_height=max_height, **kwargs);
                if(vpath):
                    num_frames_total = self.getVersionInfo(version_label=max_height, info_label='num_frames_total');
                    if(num_frames_total):
                        v = self.RegisteredVideo(path=vpath, version_label = max_height, num_frames_total=num_frames_total, load_features=True);
                    else:
                        v = self.RegisteredVideo(path=vpath, version_label=max_height, load_features=True);
                        self.setVersionInfo(version_label=max_height, num_frames_total=v.num_frames_total);
                        self.save();
                else:
                    AWARN("COULDNT GET VIDEO FROM {}".format(self.source_location));
            else:
                AWARN("COULDNT FIND VIDEO LOCALLY");
                return;
        return v;

    def saveFeaturesForVideo(self, video, features_to_save=None, output_dir=None, overwrite=True):
        if(features_to_save is None):
            return;
        if(not isinstance(features_to_save, list)):
            features_to_save=[features_to_save];
        for f in features_to_save:
            self.saveFeatureForVideo(video=video, feature_name=f, output_dir=output_dir, overwrite=overwrite);

    def saveFeatureForVideo(self, video, feature_name, output_dir=None, overwrite=True):
        if(feature_name=='each'):
            return self.saveFeaturesForVideo(video=video, features_to_save=video.getFeaturesList(), output_dir=output_dir, overwrite=overwrite);

        version_label = video.getInfo(label='version_label');
        source_type = video.getFeatureSourceType(feature_name);
        opath = self.getFeaturePath(feature_name=feature_name, version_label=version_label, source_type=source_type, output_dir=output_dir);
        make_sure_dir_exists(opath);
        if(not os.path.isfile(opath) or overwrite):
            if(feature_name=='all'):
                video.saveFeatures(path=opath);
                return;
            else:
                vfeature = video.getFeature(name=feature_name, force_recompute=False);
                if(vfeature is not None):
                    video.saveFeature(name=feature_name, path=opath);

    def loadFeaturesForVideo(self, video, features_to_load=None):
        if(features_to_load is None):
            return;
        if(not isinstance(features_to_load, list)):
            features_to_load=[features_to_load];
        for f in features_to_load:
            self.loadFeatureForVideo(video=video, feature_name=f);

    def loadFeatureForVideo(self, video, feature_name):
        if(feature_name=='each'):
            return self.loadFeaturesForVideo(video=video, features_to_load=video.getFeatureFunctionsList());
        version_label = video.getInfo(label='version_label');
        source_type=video.getFeatureSourceType(feature_name);
        ipath = self.getFeaturePath(feature_name=feature_name, version_label=version_label, source_type=source_type);
        if(os.path.isfile(ipath)):
            if(feature_name=='all'):
                video.loadFeatures(path=ipath);
            else:
                video.loadFeature(name=feature_name, path=ipath);

    def getFeaturePath(self, feature_name=None, source_type=None, version_label=None, output_dir=None):
        assert(feature_name is not None), 'must provide name of feature VideoSource.getFeaturePath'
        feature_dir = self.getFeatureDir(feature_name=feature_name, source_type = source_type);
        fileext = '.pkl';
        if(output_dir is None):
            version = self._getVersionLabelDirName(version_label=version_label)+os.sep;
            output_dir = pathstring(feature_dir);
        outname = self.getName();
        if(outname is None):
            outname = 'version';
        outname = outname+'_'+self._getVersionLabelDirName(version_label=version_label)+fileext;
        opath = os.path.join(output_dir, outname);
        return opath;

    def getFeatureDir(self, feature_name=None, source_type=None):
        if(source_type is None):
            source_type='video'; # could be 'audio'
        # AWARN("getDir('features')= {}\nsource_type= {}\nfeature_name= {}".format(self.getDir('features'), source_type, feature_name))
        ftypedir = pathstring(self.getDir('features') + os.sep + source_type + os.sep + feature_name + os.sep);
        return ftypedir;

    def saveFeaturesForVersion(self, version_label=None, output_dir=None):
        assert(output_dir), 'must provide output dir';
        v = self.getVersion(max_height=version_label);
        v.save(features_to_save='all', output_dir=output_dir);


    def RegisteredVideo(self, path=None, version_label=None, version_group = None, num_frames_total=None, load_features = True):
        v = self.VideoClass(path=path, name=self.getName()+'_'+self._versionLabelString(version_label=version_label), num_frames_total=num_frames_total);
        # v = Video(path=path, name=self.getName()+'_'+self._versionLabelString(version_label=version_label), num_frames_total=num_frames_total);
        self.RegisterVideo(video=v, version_label=version_label, load_features=load_features);
        return v;

    def RegisterVideo(self, video, version_label = None, version_group = None, load_features=True):
        def videosave(vidob, features_to_save='all', output_dir=None, overwrite=True):
            self.saveFeaturesForVideo(video=vidob, features_to_save=features_to_save, output_dir=output_dir, overwrite=overwrite);
            #self.save
        def videoload(vidob, features_to_load='all', input_dir=None):
            self.loadFeaturesForVideo(video=vidob, features_to_load=features_to_load);
        def videoclear(vidob, features_to_clear='all'):
            self.removeFeatureFilesForVideo(video=vidob, features_to_remove=features_to_clear);

        video.setInfo(label='version_label', value=version_label);
        video.setInfo(label='version_group', value=version_group);
        video.setInfo(label='video_file_name', value=os.path.split(video.getPath())[1]);
        # if(version_group is None):
        video.save_func = videosave;
        video.load_func = videoload;
        video.clear_feature_files_func = videoclear;
        video.source=self;

        if(load_features is True and hasattr(video, "loadFlowFeatures")):
            video.loadFlowFeatures();
        elif(load_features is not None and (isinstance(load_features, list) or isinstance(load_features, str))):
            video.load(features_to_load = load_features);

        # return video;

    def addVersion(self, path, version_label=None, version_group=None):
        v = self.RegisteredVideo(path=path, version_label=version_label);
        video_dict=v.getVersionInfo();
        video_dict.update(dict(version_group=version_group));
        self.setVersionInfo(video_path = path, version_label=version_label, **video_dict);
        self.save();

    def addVersionToVideo(self, video, new_video, version_label=None, version_group=None):
        version_label = video.getInfo(label='version_label');
        video_dict = new_video.getVersionInfo();
        if ((version_group is not None) or (video_dict.get('version_group') is None)):
            video_dict.update(dict(version_group=version_group));
        self.setVersionInfo(video_path=new_video.getPath(), version_label=version_label, **video_dict);
        self.save();


    def setSource(self, source_location=None, assert_valid=True, **kwargs):
        """
        Sets the source of this VideoSource. If file, copies the file to VideoSource directory as "Original" version,
            unless copy argument is set to false.
        :param source_location: either path to file, or youtube url
        :param kwargs: see setSourceYoutube or setSourceFile for options
        :return:
        """

        if(source_location):
            if(os.path.isfile(source_location)):
                self.setSourceFile(path = source_location, **kwargs);
            else:
                self.setSourceYoutube(url=source_location, **kwargs);
            return;
        elif(kwargs.get('video_path')):
            self.setSourceFile(**kwargs);
            return;
        elif(kwargs.get('url')):
            self.setSourceYoutube(**kwargs);
            return;
        else:
            if(assert_valid):
                assert(False),'No valid source location provided to setSource.'




    def setSourceYoutube(self, url=None, max_height=None, pull_fullres=True, **kwargs):
        self.source_location=url;
        self.setInfo(label='source_type', value='youtube');
        if(pull_fullres):
            self.pullYoutubeVideo(max_height=max_height, **kwargs);
        self.save();

    def setSourceFile(self, path=None, copy=True, **kwargs):
        # assert(os.path.isfile(path)), 'Tried to set source but no file exists at {}'.format(path);
        if(not os.path.isfile(path)):
            return None;

        self.video_file_name = os.path.basename(path);
        if(not copy):
            self.source_location = path;
            self.setInfo(label='source_type', value='file_address');
            return path;
        # output_dir = self.getDir('versions') + os.sep + 'Original' + os.sep;
        output_dir = self.getDirForVersion(version_label=None, version_group='Source');
        make_sure_dir_exists(output_dir);
        output_path = os.path.join(output_dir, self.video_file_name);
        # max_height = kwargs.get('max_height');

        # if(max_height is not None):
        #     # print('max_height was {}'.format(max_height));
        #     original = self.VideoClass(path=path);
        #     original.writeResolutionCopyFFMPEG(path=output_path, max_height=kwargs.get('max_height'));
        # else:
        #     shutil.copy2(path, output_path);

        shutil.copy2(path, output_path);

        # self.addVersion(path=output_path, version_label='Original');
        self.addVersion(path=output_path, version_label='Full');
        self.setInfo(label='source_type', value='file');
        self.source_location = output_path;
        self.save();
        return output_path;


    def pullVersion(self, max_height=None, **kwargs):
        # assert(self.source_location is not None), "Could not pull version; source location is None."
        if(self.source_location is None):
            return None;
        source_type = self.getInfo('source_type');
        if(source_type=='youtube'):
            return self.pullYoutubeVideo(max_height=max_height, **kwargs);
        else:
            return self.pullFileVideo(max_height=max_height, **kwargs)


    def pullFileVideo(self, max_height=None, force_recompute=None):
        original_path = self.source_location;
        assert(os.path.isfile(original_path)), 'SOURCE FILE {} IS MISSING'.format(self.source_location);
        original = self.VideoClass(path=original_path);
        output_dir = self.getDirForVersion(version_label=max_height, version_group=None);
        make_sure_dir_exists(output_dir);
        output_path = os.path.join(output_dir, self.video_file_name);
        if(max_height is None):
            if(os.path.normpath(original_path) != os.path.normpath(output_path)):
                shutil.copy2(original_path, output_path);
        else:
            original.writeResolutionCopyFFMPEG(path=output_path, max_height=max_height);
        self.addVersion(path=output_path, version_label=max_height);
        return self.getVersionPath(version_label=max_height);

    def pullYoutubeVideo(self, max_height=None, write_subtitles=False, save_youtube_info=False,
                  force_redownload=False):
        """
        Downloads video from youtube with height<=max_height. Returns path to downloaded video.
        :param max_height: maximum height of video to download
        :param write_subtitles: whether to save the subtitles
        :param save_youtube_info: whether to save the info json
        :param force_redownload: whether to re-download if video exists
        :return:
        """
        assert (not self.getInfo('source_is_file')), "tried to call pullYoutubeVideo on file source {}.".format(self.getName());

        old_vid_path = self.getVersionPath(version_label=max_height);

        if ((not force_redownload) and old_vid_path and os.path.isfile(old_vid_path)):
            AWARN("Old video version exists with path {}\nSkipping download...\n".format(old_vid_path))
            return old_vid_path;

        # output_dir = self.getDir('versions') + os.sep + self._getVersionLabelDirName(version_label=max_height) + os.sep;
        output_dir = self.getDirForVersion(version_label=max_height, version_group=None);
        make_sure_path_exists(output_dir);

        if ((not force_redownload) and self.video_file_name is not None):
            path_guess = os.path.join(output_dir, self.video_file_name);
            if (os.path.isfile(path_guess)):
                AWARN("found existing file {}\nSkipping download; set force_redownload=True to overwrite old version.".format(path_guess));
                self.addVersion(path=path_guess, version_label=max_height);
                return path_guess;


        ########Download From YouTube#########
        vidpath_template = output_dir + '%(title)s-%(id)s' + '.%(ext)s';
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': vidpath_template,
        }
        if (write_subtitles):
            ydl_opts['writesubtitles'] = True;
            ydl_opts['subtitlesformat'] = '[srt]';
        if (max_height):
            ydl_opts['format'] = 'bestvideo[height<={}][ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'.format(
                max_height);
        ydl_opts['writeinfojson'] = True;
        info_dict = None;

        AWARN("Downloading {} from {}...".format(self.getName(), self.source_location));

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(self.source_location, download=True)
            # video_source = info_dict.get("source", None)
            # video_id = info_dict.get("id", None)
            # video_title = info_dict.get('title', None)
        usedtitle = info_dict['title'].replace('"', "'").replace('|', '_').replace(':', ' -').replace('/','_').replace('?', '');
        usedfilename_withoutext = usedtitle + '-' + info_dict['id'];
        usedfilename = usedfilename_withoutext + '.' + info_dict['ext'];
        filepath = output_dir + os.sep + usedfilename;
        filepathsafe = safe_file_name(filepath);
        if (os.path.isfile(filepath)):
            os.rename(filepath, filepathsafe);
        else:
            fpath = glob.glob(output_dir + os.sep + '*' + info_dict['id'] + '.mp4');
            assert (len(fpath) == 1), "Wrong number of files for {}\nFound:\n{}".format(filepath, fpath);
            os.rename(fpath[0], filepathsafe);
            jpath = glob.glob(output_dir + os.sep + '*' + info_dict['id'] + '.info.json');
            os.rename(jpath[0], safe_file_name(usedfilename_withoutext + '.info.json'));

        print(("Saved to {}".format(filepathsafe)));

        self.video_file_name = safe_file_name(usedfilename);
        self.title_safe = safe_file_name(usedtitle);
        if (save_youtube_info):
            self.youtube_info_dict = info_dict;

        # v = Video(path=filepathsafe);
        # video_dict=v.toDictionary();
        # self.setVersionInfo(video_path = filepathsafe, version_label=max_height, num_frames_total=v.num_frames_total, **video_dict);
        # self.save();
        self.addVersion(path=filepathsafe, version_label=max_height);
        return self.getVersionPath(version_label=max_height);
        # return youtube_link;
