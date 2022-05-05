
from sys import platform
if platform == "linux" or platform == "linux2":
    PLATFORM = 'linux';
elif platform == "darwin":
    PLATFORM = 'osx';
elif platform == "win32":
    PLATFORM = 'windows'

SUPPORTED = False;

INITIAL_DIR ='./';

if(PLATFORM == 'osx'):
    from . import uipath

    def GetFilePath(initial_path=None):
        if(initial_path is None):
            return uipath.uiGetFilePath(initial_path=INITIAL_DIR);
        else:
            return uipath.uiGetFilePath(initial_path);
    def GetDirectory(initial_path=None):
        if(initial_path is None):
            return uipath.uiGetDirectory(initial_path=INITIAL_DIR);
        else:
            return uipath.uiGetDirectory(initial_path);
    def GetSaveFilePath(initial_path=None, file_extension = None):
        if(initial_path is None):
            return uipath.uiGetSaveFilePath(initial_path=INITIAL_DIR, file_extension=file_extension);
        else:
            return uipath.uiGetSaveFilePath(initial_path=initial_path, file_extension=file_extension);

    def Show(path):
        uipath.showInFinder(path);

    def Open(path):
        uipath.openOSX(path);
    SUPPORTED = True;
else:
    def GetFilePath(initial_path=None):
        return None;
    def GetDirectory(initial_path=None):
        return None;
    def GetSaveFilePath(initial_path=None, file_extension = None):
        return None;
    def Show(path):
        return None;
    def Open(path):
        return None;