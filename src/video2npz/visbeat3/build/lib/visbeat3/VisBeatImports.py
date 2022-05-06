from .VisBeatDefines import *
from .AImports import *
from .AObject import AObject
import numpy as np
import scipy as sp

import os
import imageio

import matplotlib
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    AWARN("matplotlib problem... if you are using conda try installing with 'conda install matplotlib'")
    matplotlib.use('agg');
    import matplotlib.pyplot as plt
import matplotlib.style as ms

import io
import base64
import math
from operator import truediv
import time
import shutil
from time import gmtime, strftime, localtime
import librosa;
from ._mediafiles import GetVBMarkPath

def local_time_string():
    return strftime("%Y-%m-%d_%H:%M:%S", localtime());


def VBWARN(message):
    print(message)
    #
    # def send_warnings_to_print_red(message, category, filename, lineno):
    #     print(colored('{} WARNING! file: {} Line:{}\n{}'.format(category, filename, lineno, message), 'red'))
    # old_showwarning = warnings.showwarning
    # warnings.showwarning = send_warnings_to_print_red;

VB_MACHINE_ID = None;
# if(VB_MACHINE_ID):
    # matplotlib.use('PS');

ISNOTEBOOK = False;
if(runningInNotebook()):
    ISNOTEBOOK = True;
    import IPython;
    import IPython.display
    ms.use('seaborn-muted')
    if(not runningInSpyder()):
        get_ipython().magic('matplotlib inline')
        from IPython.lib import kernel
        connection_file_path = kernel.get_connection_file()
        connection_file = os.path.basename(connection_file_path)
        kernel_id = connection_file.split('-', 1)[1].split('.')[0]
        # print("Kernel ID:\n{}".format(kernel_id));
    from IPython.display import HTML
    VBIPY = IPython;
    #%matplotlib inline
# else:
    # matplotlib.use('PS');


def vb_get_ipython():
    return VBIPY;

def clipping_params(clip_bins=30, clip_fraction=0.95):
    return dict(clip_bins=clip_bins, clip_fraction=clip_fraction);

def get_hist_clipped(signal, clip_bins=30, clip_fraction=0.95):
    holdshape = signal.shape[:];
    sigrav = signal.copy().ravel();
    sigh, sigb = np.histogram(sigrav, bins=clip_bins);
    maxbini=np.argmax(sigh);
    totalmass = np.sum(sigh);
    total_included = truediv(sigh[maxbini],totalmass);
    prevbini = maxbini;
    nextbini=maxbini;
    bins_included = 1;
    icounter=0;
    while(total_included<clip_fraction and bins_included<clip_bins):
        if((prevbini>=0 and sigh[prevbini]==0) and (nextbini<(clip_bins) and sigh[nextbini]==0)):
            prevbini=prevbini-1;
            nextbini=nextbini+1;
        else:
            if((prevbini>=0 and sigh[prevbini]>0) or nextbini>=clip_bins):
                prevbini=prevbini-1;
            if((nextbini<(clip_bins) and sigh[nextbini]>0) or prevbini<0):
                nextbini=nextbini+1;
        included_segment=sigh[max(prevbini,0):min(nextbini+1, clip_bins)];
        total_included = truediv(np.sum(included_segment), totalmass);
        bins_included=len(included_segment);
        icounter+=1;
    clipsig = np.clip(a=sigrav, a_min=sigb[max(prevbini,0)], a_max=sigb[min(nextbini,clip_bins)]);
    clipsig.shape=signal.shape
    return clipsig;

def np_scale_to_range(data, value_range=None):
    if(value_range is None):
        value_range = [0.0,255.0];
    d = data.copy().ravel();
    currentmin=np.min(d);
    currentmax=np.max(d);
    currentscale = currentmax-currentmin;
    if(currentscale==0):
        VBWARN("CANNOT SCALE CONSTANT ARRAY TO RANGE")
        return;
    divscale = truediv(1.0,currentscale);
    newrange=(value_range[1]-value_range[0]);
    d = (d*divscale)*newrange;
    newmin = (currentmin*divscale)*newrange;
    d = d-newmin+value_range[0]
    d.shape=data.shape;
    return d


