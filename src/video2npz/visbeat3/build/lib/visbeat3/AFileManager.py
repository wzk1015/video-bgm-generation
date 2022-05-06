from .AObject import *
#import shutil
from distutils.dir_util import copy_tree

class AFileManager(AObject):
    """AFileManager (class): Manages assets. This should really be replaced with a database of some sort...
        Attributes:
            todo
    """

    @staticmethod
    def AOBJECT_TYPE():
        return 'AFileManager';

    def getJSONPath(self):
        return self.getPath();

    def __init__(self, path=None, clear_temp=None):
        """If you provide a directory, it will look for a existing AFileManager.json in that directory, or create one if it does not already exist.
            If you provide a json, it will use that json, unless the json doesn't exist, in which case it will complain...
        """
        AObject.__init__(self, path=path);
        # self.initializeBlank();
        self.initWithPath(path=path, clear_temp=clear_temp);

    def initializeBlank(self):
        AObject.initializeBlank(self);
        self.directories = {};

    def getJSONName(self):
        return self.AOBJECT_TYPE()+".json";

    def initWithPath(self, path=None, clear_temp=None):
        oldpath = None
        newpath = path;
        if(path):
            if(os.path.isfile(path)):
                self.loadFromJSON(self.getJSONPath()); #assume path property is already set to 'path'
                oldpath = self.getPath(); #whatever was in the json, having overwritten path property
            elif(os.path.isdir(path)):
                json_file_path = path+os.sep+self.getJSONName();
                self.setPath(json_file_path);
                if(os.path.isfile(self.getJSONPath())):
                    self.loadFromJSON(json_file_path);
                    oldpath = self.getPath();
                    newpath = json_file_path;
                    # self.setPath(file_path=json_file_path);
                else:
                    newpath=self.getJSONPath()
                    self.writeToJSON(json_path=newpath);#no json file found, so we create one
            else:
                assert False, "Given AFileManager path is neither an existing directory or file! path: {} (AFileManager.py)".format(path)

            self.setPath(file_path=newpath);

            if(oldpath):
                oldir = get_dir_from_path(pathstring(oldpath));
                newdir = get_dir_from_path(pathstring(newpath));
                if(oldir != newdir):
                    AWARN("FILEMANAGER FOUND FILE MOVED FROM:\n{}\nTO:\n{}\nUPDATING DIRECTORIES...".format(oldir,
                                                                                                            newdir));
                    for d in self.directories:
                        dpth = self.directories[d];
                        if(dpth.startswith(oldir)):
                            dpthst = dpth.lstrip(oldir);
                            self.directories[d]=os.path.join(newdir,dpthst);
                            AWARN("{} updated to {}".format(dpth, self.directories[d]));


            self.setDir('data', pathstring(self.getDirectoryPath()+os.sep+"Data"+os.sep));
            self.setDir('backup', pathstring(self.getDir('data')+"Backups"+os.sep));
            self.setDir('temp', pathstring(self.getDir('data')+"TEMP"+os.sep));
            temp_dir = self.getDir('temp');
            if(os.path.isdir(temp_dir) and (clear_temp)):
                for the_file in os.listdir(temp_dir):
                    file_path = os.path.join(temp_dir, the_file);
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path);
                            #os.unlink(file_path);
                            #elif os.path.isdir(file_path): shutil.rmtree(file_path)
                    except Exception as e:
                        print(e)
            make_sure_path_exists(temp_dir);
            #Video.VIDEO_TEMP_DIR = temp_dir;

    def setDir(self, name, path):
        # AWARN("setting {} to {}".format(name, path))
        # assert(name is not 'log')
        self.directories[name]=path;
        make_sure_path_exists(path);
        return path;

    def addDir(self, name):
        assert(name not in self.directories), "tried to add {} dir to AFileManager, but this dir is already set"
        return self.setDir(name, pathstring(self.getDirectoryPath()+os.sep+name+os.sep));

    def getDir(self, name):
        # printDictionary(self.directories)
        return self.directories.get(name);


    def emptyDir(self, name):
        dpth = self.getDir(name);
        if(dpth is not None and os.path.isdir(dpth)):
            shutil.rmtree(dpth);
            make_sure_path_exists(dpth);

    def deleteDir(self, name):
        dpth = self.getDir(name);
        if (dpth is not None and os.path.isdir(dpth)):
            shutil.rmtree(dpth);
            d = dict(self.directories);
            del d[name];
            self.directories=d;


    def toDictionary(self):
        d = AObject.toDictionary(self);
        d['directories']=self.directories;
        #serialize class specific members
        return d;

    def copyPathToDir(self, path_to_copy, dest_dir):
        dest_path = self.getDir(dest_dir);
        if(dest_path):
            if(os.path.isdir(path_to_copy)):
                copy_tree(src=path_to_copy, dst=dest_path);
            elif(os.path.isfile(path_to_copy)):
                shutil.copy2(path_to_copy, dest_path)
        return;

    def copyDirToPath(self, dir_to_copy, dest_path):
        src_path = self.getDir(dir_to_copy);
        if(src_path):
            if(os.path.isdir(dest_path)):
                copy_tree(src=src_path, dst=dest_path);
        return;

    @staticmethod
    def copyRandomFractionOfFilesInSourceDir(source_dir, dest_dir, fraction=1.0, ext=None):
        """
        Copies a random fraction of files in source directory... Wrote this for splitting training/test data in ML applications.
        :param source_dir:
        :param dest_dir:
        :param fraction:
        :param ext:
        :return:
        """
        directories = []
        subdirnames = []
        filepaths = [];
        for filename in os.listdir(source_dir):
            path = os.path.join(source_dir, filename)
            if os.path.isdir(path):
                directories.append(path)
                subdirnames.append(filename)
            else:
                # namepart, extpart = os.path.splitext(filename);
                if((ext is None) or filename.lower().endswith(ext)):
                    filepaths.append(path);

        n_to_copy = int(len(filepaths)*fraction);
        random_seed = 0;
        random.seed(random_seed);
        random.shuffle(filepaths);
        copy_sources = filepaths[:n_to_copy];
        for src, dst in zip(copy_sources, [dest_dir]*len(copy_sources)):
            #print("src: {}\ndst: {}".format(src, dst));
            shutil.copy2(src, dst);

        for d in range(len(directories)):
            subdest = pathstring(os.path.join(dest_dir,subdirnames[d])+os.sep);
            make_sure_dir_exists(subdest);
            AFileManager.copyRandomFractionOfFilesInSourceDir(source_dir=directories[d], dest_dir=subdest, fraction=fraction, ext=ext);

    def initFromDictionary(self, d):
        AObject.initFromDictionary(self, d);
        self.directories = d['directories'];

    def save(self):
        if(os.path.isfile(self.getJSONPath())):
            os.rename(self.getJSONPath(), self.getDir('backup')+os.sep+self.AOBJECT_TYPE()+".json");
        self.writeToJSON(self.getJSONPath());
