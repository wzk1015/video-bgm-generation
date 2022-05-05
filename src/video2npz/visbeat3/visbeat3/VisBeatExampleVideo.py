import os
from visbeat3.SourceLocationParser import ParseSourseLocation

class VisBeatExampleVideo(object):
    def __init__(self, name, url, start_beat=None, end_beat = None, display_name = None, code=None, leadin=None, **kwargs):
        self.name = name;
        self.url = url;
        self.start_beat = start_beat;
        self.end_beat = end_beat;
        self._display_name = display_name;
        self._code = code;
        self.leadin = leadin;
        for k in kwargs:
            setattr(self, k, kwargs[k]);
        if(code is None):
            sloc = ParseSourseLocation(self.url);
            self._code = sloc.code;
    # <editor-fold desc="Property: 'code'">
    @property
    def code(self):
        return self._getCode();
    def _getCode(self):
        return self._code;
    @code.setter
    def code(self, value):
        self._setCode(value);
    def _setCode(self, value):
        self._code = value;
    # </editor-fold>

    # def _ytEmbedCode(self):

    # <editor-fold desc="Property: 'display_name'">
    @property
    def display_name(self):
        return self._getDisplayName();
    def _getDisplayName(self):
        if(self._display_name is None):
            return self.name;
        else:
            return self._display_name;
    # </editor-fold>


    def _ytWatchURL(self):
        return 'https://www.youtube.com/watch?v={}'.format(self.code);

    def _ytEmbedURL(self, autoplay=None):
        s = 'https://www.youtube.com/embed/{}'.format(self.code);
        if(autoplay):
            s = s+'?autoplay=1';
        return s;

    def _ytThumbURL(self):
        return 'https://ytimg.googleusercontent.com/vi/{}/default.jpg'.format(self.code);

    def _fancyBoxCode(self, with_label = None):
        """"""
        s = HTMLCode();
        if(with_label):
            s.addLine("<figure>")
        s.addLine('<a class="vb_youtube_fancybox" data-fancybox href="{watchurl}"><img alt="{alt_name}" src="{thumburl}" /></a>'.format(watchurl=self._ytWatchURL(),
                                                                                                                                        alt_name = self.display_name,
                                                                                                                                        thumburl = self._ytThumbURL()))

        if(with_label):
            s.addLine('<figcaption>');
            # s.addLine('<a href="{watchurl}">{displayname}</a>'.format(self.url, self.display_name));
            s.addLine('{displayname}'.format(displayname=self.display_name));
            s.addLine('</figcaption>')
            s.addLine('</figure>');
        return s.string;


from bs4 import BeautifulSoup
class HTMLCode(object):
    def __init__(self, start_string=None):
        self._code = "";
        self._lines = [];
        if(start_string is not None):
            self.add(start_string)
        self._soup = None;

    # <editor-fold desc="Property: 'string'">
    @property
    def string(self):
        return self._getString();
    def _getString(self):
        return BeautifulSoup(self.code).prettify();

    # <editor-fold desc="Property: 'code'">
    @property
    def code(self):
        return self._getCode();
    def _getCode(self):
        return self._code;
    @code.setter
    def code(self, value):
        self._setCode(value);
    def _setCode(self, value):
        self._code = value;
    # </editor-fold>


    # <editor-fold desc="Property: 'soup'">
    @property
    def soup(self):
        return self._getSoup();
    def _getSoup(self):
        if(self._soup is None):
            self._makeSoup();
        return self._soup;

    def _makeSoup(self):
        self._soup = BeautifulSoup(self.code, 'html.parser');
    # @soup.setter
    # def soup(self, value):
    #     self._setSoup(value);
    # def _setSoup(self, value):
    #     self._soup = value;
    # </editor-fold>

    def add(self, s):
        self.code = self.code+s;

    def addLine(self, s):
        self.code = self.code+'\n'+s;

    def startTable(self, id=None, class_ = None, **kwargs):
        targs = dict(table_width="80%", border=1, cellspacing=1, cellpadding=1)
        targs.update(kwargs);
        s = '<table ';
        if(id is not None):
            s = s+'id="{}" '.format(id);
        if(class_ is not None):
            s = s + 'class="{}" '.format(class_);
        s = s+'width="{table_width}" border="{border}" cellspacing="{cellspacing}" cellpadding="{cellpadding}">'.format(**targs);
        self.addLine(s);
        self.addLine('<tbody>');

    def endTable(self):
        self.addLine("</tbody>");
        self.addLine("</table>");

    def startRow(self):
        self.addLine('<tr>');

    def endRow(self):
        self.addLine('</tr>');

    def addColumnLabel(self, label):
        self.addLine('<th scope="col">')
        self.addLine(label);
        self.addLine('</th>');

    def addRowLabel(self, label):
        self.addLine('<th scope="row">')
        self.addLine(label);
        self.addLine('</th>');

    def addRowCell(self, content):
        self.addLine("<td>");
        self.addLine("<div class = 'rowcelldiv'>")
        self.addLine(content);
        self.addLine("</div>")
        self.addLine("</td>");
