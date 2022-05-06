import os


VB_MEDIA_UTILS_PATH = os.path.abspath(__file__)
VB_MEDIA_UTILS_DIR = os.path.abspath(os.path.dirname(__file__));
MEDIAFILES_DIR = os.path.join(VB_MEDIA_UTILS_DIR, 'assets'+os.sep)

AUDIO_FILES_DIR = os.path.join(MEDIAFILES_DIR, 'audio'+os.sep);
AUDIO_FILES = [];
AUDIO_FILE_PATHS = {};
for filename in os.listdir(AUDIO_FILES_DIR):
    # if(reduce(lambda x,y: x or y, map(lambda ext: filename.lower().endswith(ext), Audio.MEDIA_FILE_EXTENSIONS()))):
    AUDIO_FILES.append(filename);
    AUDIO_FILE_PATHS[filename]=(os.path.join(AUDIO_FILES_DIR, filename));


def GetTestAudioPath(filename):
    return AUDIO_FILE_PATHS[filename];


VIDEO_FILES_DIR = os.path.join(MEDIAFILES_DIR, 'video'+os.sep);
VIDEO_FILES = [];
VIDEO_FILE_PATHS = [];
if(os.path.exists(VIDEO_FILES_DIR)):
    for filename in os.listdir(VIDEO_FILES_DIR):
        VIDEO_FILES.append(filename);
        VIDEO_FILE_PATHS.append(os.path.join(VIDEO_FILES_DIR, filename));



IMAGE_FILES_DIR = os.path.join(MEDIAFILES_DIR, 'images'+os.sep);
IMAGE_FILES = [];
IMAGE_FILE_PATHS = {};
for filename in os.listdir(IMAGE_FILES_DIR):
    IMAGE_FILES.append(filename);
    IMAGE_FILE_PATHS[filename] = (os.path.join(IMAGE_FILES_DIR, filename));

def GetTestImagePath(filename=None):
    if(filename is None):
        filename = "VisBeatWatermark.png"
    return IMAGE_FILE_PATHS[filename];

def GetVBMarkPath():
    return IMAGE_FILE_PATHS["VisBeatWatermark.png"];
