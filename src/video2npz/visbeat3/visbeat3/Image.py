#Image_Base
#import numpy as np
#import scipy as sp
from PIL import Image as PIM
from PIL import ImageDraw, ImageFont

from .VisBeatImports import *
# from PlotUtils import *


VBMARK_SIZE = 0.135;
VBMMARGIN = 0.02;

def imshow(imdata, new_figure = True):
    if(ISNOTEBOOK):
        #plt.imshow(self.data*0.0039215686274509);#divided by 255
        if(new_figure):
            plt.figure();
        if(len(imdata.shape)<3):
            plt.imshow(imdata, cmap='gray');
        else:
            plt.imshow(imdata);#divided by 255
        plt.axis('off');


class Image(AObject):
    """Image
    """
    IMAGE_TEMP_DIR = None;
    USING_OPENCV=False;
    _VBMRK = None;

    def AOBJECT_TYPE(self):
        return 'Image';

    def __init__(self, data=None, path = None):
        AObject.__init__(self, path=path);
        self.data = data;
        if(path and (not data)):
            self.loadImageData();

    def initializeBlank(self):
        AObject.initializeBlank(self);
        self.data = None;

    def setBlank(self, shape=None):
        if(not shape):
            self.data = np.zeros(self.data.shape);
        else:
            self.data = np.zeros([shape[0], shape[1], ]);

    def loadImageData(self, path = None, force_reload=True):
        if(path):
            self.setPath(path);
        if(self.a_info.get('file_path')):
            if(force_reload or (not self.data)):
                pim = PIM.open(fp=self.a_info['file_path']);
                self.data = np.array(pim);

    @property
    def shape(self):
        return self._getShape();

    def _getShape(self):
        return np.asarray(self.data.shape)[:];

    @property
    def dtype(self):
        return self.data.dtype;

    @property
    def width(self):
        return self._getWidth();

    def _getWidth(self):
        return self.shape[1];

    @property
    def height(self):
        return self._getHeight();

    def _getHeight(self):
        return self.shape[0];

    @property
    def _is_float(self):
        return (self.dtype.kind in 'f');

    @property
    def _is_int(self):
        return (self.dtype.kind in 'iu');

    @property
    def _pixels_float(self):
        if (self._is_float):
            return self.data;
        else:
            return self.data.astype(np.float) * np.true_divide(1.0, 255.0);

    @property
    def _pixels_uint(self):
        if (self._is_int):
            return self.data;
        else:
            return (self.data * 255).astype(np.uint8);

    @property
    def n_channels(self):
        return self._getNChannels();

    def _getNChannels(self):
        if(len(self.data.shape)<3):
            return 1;
        else:
            return self.data.shape[2];

    @staticmethod
    def FromGrayScale(gray_data, color_map = None, format=None, **kwargs):
        if(not color_map):
            newim = Image(data=np.dstack((gray_data,gray_data,gray_data)));
        else:
            cmap=matplotlib.cm.get_cmap(name=color_map);
            #norm = mpl.colors.Normalize(vmin=-20, vmax=10)
            #cmapf = cm.ScalarMappable(norm=norm, cmap=cmap)
            cmapf = matplotlib.cm.ScalarMappable(cmap=cmap);
            if(format is None):
                newim = Image(data=cmapf.to_rgba(gray_data)*255);
            else:
                assert(False), "unrecognized format {}".format(format);

        newim.a_info['ColorType']='FromGrayScale';
        return newim;

    @staticmethod
    def printTempDir():
        print((Image.IMAGE_TEMP_DIR));

    def scaleToValueRange(self, value_range=None):
        is_int = self._is_int;
        if(value_range is None):
            value_range = [0.0,255.0];
        data = self.data.ravel();
        currentmin=np.min(data);
        currentmax=np.max(data);
        currentscale = currentmax-currentmin;
        if(currentscale==0):
            VBWARN("CANNOT SCALE CONSTANT IMAGE")
            return;
        divscale = truediv(1.0,currentscale);
        newrange=(value_range[1]-value_range[0]);
        self.data = (self.data*divscale)*newrange;
        newmin = (currentmin*divscale)*newrange;
        self.data = self.data-newmin+value_range[0]
        if(is_int and value_range[1]>254):
            self.data = self.data.astype(np.int);


    def nChannels(self):
        if(len(self.data.shape)<3):
            return 1;
        else:
            return self.data.shape[2];

    def getClone(self):
        rimg = Image(path=self.a_info.get('file_path'), data=self.data.copy());
        return rimg;

    def getGridPixel(self,x,y,repeatEdge=0):
        xo = x;
        yo = y;

        blackedge = np.zeros(1);
        if(len(self.data.shape)==3):
            blackedge = np.zeros(self.data.shape[2])

        if(y>=self.data.shape[0]):
            if(repeatEdge==1):
                yo = self.data.shape[0]-1
            else:
                return blackedge

        if(y<0):
            if(repeatEdge==1):
                yo = 0
            else:
                return blackedge

        if(x>=self.data.shape[1]):
            if(repeatEdge==1):
                xo=self.data.shape[1]-1
            else:
                return blackedge

        if(x<0):
            if(repeatEdge==1):
                xo = 0
            else:
                return blackedge

        return self.data[yo,xo]

    def getPixel(self, x, y, repeatEdge=0):
        if(isinstance(y,int) and isinstance(x,int)):
            return self.getGridPixel(x,y)
        else:
            yf = int(np.floor(y))
            yc = int(np.ceil(y))
            xf = int(np.floor(x))
            xc = int(np.ceil(x))

            print('getting here?')
            print(xf);
            print(yf);

            tl = self.getGridPixel(xf,yf,repeatEdge)
            tr = self.getGridPixel(xc,yf,repeatEdge)
            bl = self.getGridPixel(xf,yc,repeatEdge)
            br = self.getGridPixel(xc,yc,repeatEdge)

            yalpha = y-yf
            xalpha = x-xf

            topL = tr*xalpha+tl*(1.0-xalpha)
            botL = br*xalpha+bl*(1.0-xalpha)

            retv = botL*yalpha+topL*(1.0-yalpha)
            return retv

    def getShape(self):
        return np.asarray(self.data.shape)[:];

    def getScaled(self, shape=None, shape_xy=None):
        shapeis = (shape is not None);
        shapexyis = (shape_xy is not None);
        assert((shapeis or shapexyis) and not (shapeis and shapexyis)), "Must provide only one of shape or shape_xy for Image.getScaled"
        if(shapeis):
            sz=[shape[0], shape[1], self.data.shape[2]];
        else:
            sz=[shape_xy[1],shape_xy[0],self.data.shape[2]];
        imK = sp.misc.imresize(self.data, size=sz);
        return Image(data=imK);

    def getRotated(self, theta):
        imR = sp.misc.imrotate(self.data, theta);
        return Image(data=imR);


    # def splatAtPixCoord(self, im, location=[0,0]):
    #     self.data[location[0]:(location[0]+im.data.shape[0]), location[1]:(location[1]+im.data.shape[1]),:]=im.data;
    def _splatAtPixCoord(self, im, location=[0,0], **kwargs):
        is_int = self._is_int;
        selftype = self.dtype;
        region0 = [location[0], (location[0]+im.data.shape[0])];
        region1 = [location[1], (location[1]+im.data.shape[1])];
        if(im.n_channels<4):
            self.data[region0[0]:region0[1], region1[0]:region1[1],:]=im.data;
            return;
        if(im.n_channels == 4):
            alphamap = np.moveaxis(np.tile(im._pixels_float[:, :, 3], (3, 1, 1)), [0], [2]);

            blenda = (im._pixels_float[:, :, :3]) * alphamap + self._pixels_float[region0[0]:region0[1], region1[0]:region1[1],:]*(1.0 - alphamap);
            if(is_int):
                blenda = (blenda*255).astype(selftype);
            self.data[region0[0]:region0[1], region1[0]:region1[1], :] = blenda;



    def reflectY(self):
        self.data[:,:,:]=self.data[::-1,:,:];

    def reflectX(self):
        self.data[:,:,:]=self.data[:,::-1,:];

    def PIL(self):
        return PIM.fromarray(np.uint8(self.data));

    def getRGBData(self):
        return self.data[:,:,0:3];

    def normalize(self, scale=1.0):
        self.data = self.data/np.max(self.data.ravel());
        self.data = self.data*scale;

    def show(self, new_figure = True):
        if(ISNOTEBOOK):
            if(new_figure):
                plt.figure();
            if(self.nChannels()==1):
                plt.imshow(self.PIL(), cmap='gray');
            else:
                plt.imshow(self.PIL());#divided by 255
            plt.axis('off');
        else:
            self.PIL().show();

    def writeToFile(self, out_path):
        self.PIL().save(out_path);

    def getEncodedBase64(self):
        return base64.b64encode(self.data);

    def getDataAsString(self):
        return self.data.tostring();

    @staticmethod
    def FromBase64(encoded_data, shape):
        d = base64.decodestring(encoded_data);
        npar = np.frombuffer(d, dtype=np.float64);
        rIm = Image(data=np.reshape(npar, shape));
        return rIm;




    @staticmethod
    def FromDataString(data_string, shape, dtype=None):
        if(dtype is None):
            dtype=np.float64;
        img_1d = np.fromstring(data_string, dtype=dtype);
        reconstructed_img = img_1d.reshape((height, width, -1))


###########################adapted from https://gist.github.com/turicas/1455973##########################
    def _get_font_size(self, text, font_path, max_width=None, max_height=None):
        if max_width is None and max_height is None:
            raise ValueError('You need to pass max_width or max_height')
        font_size = 1
        text_size = self.get_text_size(font_path, font_size, text)
        if (max_width is not None and text_size[0] > max_width) or (max_height is not None and text_size[1] > max_height):
            raise ValueError("Text can't be filled in only (%dpx, %dpx)" % text_size)
        while True:
            if (max_width is not None and text_size[0] >= max_width) or (max_height is not None and text_size[1] >= max_height):
                return font_size - 1
            font_size += 1
            text_size = self.get_text_size(font_path, font_size, text)

    def writeOutlinedText(self, xy, text,
                   font_size=11,
                   max_width=None,
                   max_height=None,
                   encoding='utf8', draw_context = None):
        self.writeText(xy=[xy[0]+3, xy[1]+3], text=text, font_filename='RobotoCondensed-Regular.ttf',
                       font_size=font_size,
                       max_width=max_width,
                       color=(0, 0, 0),
                       max_height=max_height,
                       encoding=encoding, draw_context=draw_context);
        self.writeText(xy=xy, text=text, font_filename='RobotoCondensed-Regular.ttf',
                       font_size=font_size,
                       max_width=max_width,
                       color=(255, 255, 255),
                       max_height=max_height,
                       encoding=encoding, draw_context=draw_context);


    def writeText(self, xy, text, font_filename='RobotoCondensed-Regular.ttf',
                   font_size=11,
                   color=(0, 0, 0),
                   max_width=None,
                   max_height=None,
                   encoding='utf8', draw_context = None):

        x=xy[0];
        y=xy[1];

        font_paths = find_all_files_with_name_under_path(name=font_filename,
                                                         path=os.path.dirname(os.path.abspath(__file__)));
        font_path = font_paths[0];

        if isinstance(text, str):
            text = text.decode(encoding)
        if font_size == 'fill' and (max_width is not None or max_height is not None):
            font_size = self._get_font_size(text, font_path, max_width,
                                           max_height)
        text_size = self._get_text_size(font_path, font_size, text)


        font = ImageFont.truetype(font=font_path, size=font_size);

        # font = ImageFont.truetype(font_filename, font_size)
        if x == 'center':
            x = (self.data.shape[1] - text_size[0]) / 2
        if y == 'center':
            y = (self.data.shape[0] - text_size[1]) / 2

        if(draw_context is None):
            ipil = self.PIL();
            draw = ImageDraw.Draw(ipil)
            draw.text((x, y), text, font=font, fill=color)
            datashape = self.data.shape;
            self.data = np.array(ipil.getdata());
            self.data.shape = datashape;
        else:
            draw_context.text((x, y), text, font=font, fill=color);
        return text_size

    def _get_text_size(self, font_path, font_size, text):
        font = ImageFont.truetype(font_path, font_size)
        return font.getsize(text)

    @staticmethod
    def _VBMark():
        if(Image._VBMRK is None):
            Image._VBMRK = Image(path=GetVBMarkPath());
        return Image._VBMRK;

    def _vbmark(self):
        self._splatAtPixCoord(**self._vbmarker());

    def _vbmarker(self):
        vbm = Image._VBMark();
        wmfrac = min(VBMARK_SIZE * self.width, VBMARK_SIZE * self.height);
        scaleval = min(1.0, np.true_divide(wmfrac, vbm.width));
        if (scaleval < 1.0):
            vbms = vbm.getScaled(shape=[int(vbm.shape[0] * scaleval), int(vbm.shape[1] * scaleval), vbm.shape[2]]);
        else:
            vbms = vbm.clone();
        margin = min(vbm.width, vbm.height, int(VBMMARGIN * self.width), int(VBMMARGIN * self.height));

        return dict(im=vbms, location=[self.height - vbms.height - margin, self.width - vbms.width - margin]);

    def writeTextBox(self, xy, text, box_width, font_filename='RobotoCondensed-Regular.ttf',
                       font_size=11, color=(0, 0, 0), place='left',
                       justify_last_line=False):

        x = xy[0];
        y = xy[1];

        font_paths = find_all_files_with_name_under_path(name=font_filename,
                                                         path=os.path.dirname(os.path.abspath(__file__)));
        font_path = font_paths[0];

        lines = []
        line = []
        words = text.split()
        for word in words:
            new_line = ' '.join(line + [word])
            size = self._get_text_size(font_path, font_size, new_line)
            text_height = size[1]
            if size[0] <= box_width:
                line.append(word)
            else:
                lines.append(line)
                line = [word]
        if line:
            lines.append(line)
        lines = [' '.join(line) for line in lines if line]
        height = y
        for index, line in enumerate(lines):
            height += text_height
            if place == 'left':
                self.writeText((x, height), line,
                                font_filename,
                                font_size,
                                color)
            elif place == 'right':
                total_size = self._get_text_size(font_path, font_size, line)
                x_left = x + box_width - total_size[0]
                self.writeText((x_left, height), line, font_filename,
                                font_size, color)
            elif place == 'center':
                total_size = self._get_text_size(font_path, font_size, line)
                x_left = int(x + ((box_width - total_size[0]) / 2))
                self.writeText((x_left, height), line,
                                font_filename,
                                font_size, color)
            elif place == 'justify':
                words = line.split()
                if (index == len(lines) - 1 and not justify_last_line) or len(words) == 1:
                    self.writeText((x, height), line,
                                    font_filename,
                                    font_size,
                                    color)
                    continue
                line_without_spaces = ''.join(words)
                total_size = self._get_text_size(font_path, font_size,
                                                line_without_spaces)
                space_width = (box_width - total_size[0]) / (len(words) - 1.0)
                start_x = x
                for word in words[:-1]:
                    self.writeText((start_x, height), word,
                                    font_filename,
                                    font_size, color)
                    word_size = self._get_text_size(font_path, font_size,
                                                   word)
                    start_x += word_size[0] + space_width
                last_word_size = self._get_text_size(font_path, font_size,
                                                    words[-1])
                last_word_x = x + box_width - last_word_size[0]
                self.writeText((last_word_x, height), words[-1],
                                font_filename,
                                font_size, color)
        return (box_width, height - y)
#####################################################

from . import Image_CV
if(Image_CV.USING_OPENCV):
    Image.USING_OPENCV = Image_CV.USING_OPENCV;
    Image.RGB2Gray=Image_CV.RGB2Gray;
    Image.GrayToRGB=Image_CV.Gray2RGB;
    Image.cvGoodFeaturesToTrack=Image_CV.cvGoodFeaturesToTrack;
    Image.withFlow=Image_CV.withFlow;
    cvDenseFlowFarneback = Image_CV.cvDenseFlowFarneback;
    # Image. = Image_CV.
    # Image. = Image_CV.
    # Image. = Image_CV.
    # Image. = Image_CV.
    # Image. = Image_CV.
    ocv = Image_CV.ocv;