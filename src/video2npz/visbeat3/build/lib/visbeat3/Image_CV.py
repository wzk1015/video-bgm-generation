

#CVFunctions
from .VisBeatImports import *
import numpy as np
import scipy as sp
from PIL import Image as PIM
#from VBObject import *
from . import Image as vbImage
import math

DEFAULT_FLOW_HISTOGRAM_BINS = 8;


USING_OPENCV=True
try:
    import cv2 as ocv
except ImportError:
    USING_OPENCV=False;
    AWARN('OpenCV not installed; not importing Image_CV')


#these are functions that use OpenCV that aren't class functions
if(USING_OPENCV):
    def flow2rgb(flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
        #v=np.minimum(v*4, 255);
        v = np.log(v+1);
        vmax = np.max(v[:]);
        vf=v/vmax;
        v=vf*255;
        hsv[...,2] = v;#np.minimum(v*4, 255)
        rgb = ocv.cvtColor(hsv, ocv.COLOR_HSV2BGR)
        return rgb

    def showFlowHSV(flow, new_figure = True):
        if(ISNOTEBOOK):
            #plt.imshow(self.data*0.0039215686274509);#divided by 255
            if(new_figure):
                plt.figure();
            plt.imshow(PIM.fromarray(flow2rgb(flow)));
            plt.axis('off');
        else:
            self.PIL().show();

    def cornerHarris(im, blockSize=None, ksize=None, k=None):
        if (blockSize is None):
            blockSize = 2;
        if (ksize is None):
            ksize = 3;
        if (k is None):
            k = 0.04;
        return ocv.cornerHarris(im, 2, 3, 0.04);

    def cvDenseFlowFarneback(from_image, to_image, pyr_scale=None, levels=None, winsize=None, iterations=None, poly_n=None, poly_sigma=None, flags=None):
        """Can provide numpy arrays or Image objects as input. Returns the flow from->to."""
        # params for Farneback's method
        from_im=from_image;
        to_im=to_image;
        from_is_ob = isinstance(from_image, vbImage.Image);
        to_is_ob = isinstance(to_image, vbImage.Image)

        if(from_is_ob):
            if(from_image.nChannels()>1):
                from_im=from_image.getClone();
                from_im.RGB2Gray();
            from_im=from_im.data;
        if(to_is_ob):
            if(to_image.nChannels()>1):
                to_im=to_image.getClone();
                to_im.RGB2Gray();
            to_im=to_im.data;



        inputs = dict( flow = None,
                            pyr_scale = pyr_scale,
                            levels = levels,
                            winsize = winsize,
                            iterations = iterations,
                            poly_n = poly_n,
                            poly_sigma = poly_sigma,
                            flags=flags);

        default_flow=None;
        default_pyr_scale=0.5;
        default_levels=int(math.log(float(min(to_im.shape)), 2))-4;
        default_winsize = 15;
        default_iterations=3;
        default_poly_n=5;
        default_poly_sigma=1.25;

        use_params = dict(flow=default_flow,
                          pyr_scale=default_pyr_scale,
                          levels=default_levels,
                          winsize=default_winsize,
                          iterations=default_iterations,
                          poly_n=default_poly_n,
                          poly_sigma=default_poly_sigma,
                          flags=0);

        for key in inputs:
            if(inputs[key] is not None):
                use_params[key]=inputs[key];

        return ocv.calcOpticalFlowFarneback(prev=from_im, next=to_im, **use_params);



# These functions are to add to Image class
if(USING_OPENCV):
    def RGB2Gray(self):
        if(self.nChannels()==1):
            return;
        else:
            self.data = ocv.cvtColor(self.data, ocv.COLOR_RGB2GRAY);

    def Gray2RGB(self):
        if(self.nChannels()==1):
            self.data = ocv.cvtColor(self.data, ocv.COLOR_GRAY2RGB)

    def cvGoodFeaturesToTrack(self, maxCorners=None, qualityLevel=None, minDistance=None, corners=None, mask=None, blockSize=None, useHarrisDetector=None):
        assert(self.data is not None), "must provide image to find features";

        if(self.nChannels()==1):
            gray_image=self.data.copy();
        else:
            gray_image=ocv.cvtColor(self.data, cv2.COLOR_RGB2GRAY);

        argd={};
        if(maxCorners is not None):
            d['maxCorners']=maxCorners;
        if(qualityLevel is not None):
            d['qualityLevel']=qualityLevel;
        if(minDistance is not None):
            d['minDistance']=minDistance;
        if(corners is not None):
            d['corners']=corners;
        if(mask is not None):
            d['mask']=mask;
        if(blockSize is not None):
            d['blockSize']=blockSize;
        if(useHarrisDetector is not None):
            d['useHarrisDetector']=useHarrisDetector;
        return ocv.goodFeaturesToTrack(gray_image, **argd);



    def withFlow(self, flow, step=16):
        clone = self.getClone();
        if(clone.nChannels()<3):
            clone.convertToRGB();
        img = clone.data;
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        # flow[y, x] is fx, fy
        fx, fy = flow[y,x].T
        # After vstack, each column: x, y, x+fx, y+fy
        # After transpose: each row: x, y, x+fx, y+fy
        # After reshape: each row: [[x, y], [x+fx, y+fy]]
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        ocv.polylines(img, lines, 0, (0, 255, 0), thickness=3)
        for (x1, y1), (x2, y2) in lines:
            cv2.circle(img, (x1, y1), 1, (0, 255, 0), -1)
        return clone

    
