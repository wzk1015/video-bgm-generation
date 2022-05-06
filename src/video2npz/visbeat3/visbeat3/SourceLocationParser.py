"""
This code was a last minute hack. It works fine enough for parsing youtube urls, but I would use it for any kind of reference.
Based loosely on video backends from django embed video.
"""

import re
import requests
import os
import sys
if sys.version_info.major == 3:
    import urllib.parse as urlparse
else:
    import urllib.parse

class SourceURL(object):
    """
    Modified from django embed video
    """


    allow_https = True
    is_secure = False

    @classmethod
    def SourceLocationType(cls):
        return cls.__name__;

    def __init__(self, source_location):
        self.source_location_type = self.SourceLocationType()
        self._source_location = source_location

    # <editor-fold desc="Property: 'code'">
    @property
    def code(self):
        """
        unique string to identify file
        :return:
        """
        return self._getCode();

    def _getCode(self):
        raise NotImplementedError;

    @code.setter
    def code(self, value):
        self._setCode(value);
    def _setCode(self, value):
        raise NotImplementedError;
    # </editor-fold>


    @property
    def url(self):
        """
        URL of video.
        """
        return self.get_url()
    def get_url(self):
        raise NotImplementedError

    @property
    def protocol(self):
        """
        Protocol used to generate URL.
        """
        return 'https' if self.allow_https and self.is_secure else 'http'

    @property
    def thumbnail(self):
        """
        URL of video thumbnail.
        """
        return self.get_thumbnail_url()

    @classmethod
    def is_valid(cls, url):
        return True if cls.re_detect.match(url) else False

class WebSourceException(Exception):
    """ Parental class for all embed_media exceptions """
    pass


class VideoDoesntExistException(WebSourceException):
    """ Exception thrown if video doesn't exist """
    pass


class UnknownBackendException(WebSourceException):
    """ Exception thrown if video backend is not recognized. """
    pass


class UnknownIdException(VideoDoesntExistException):
    """
    Exception thrown if backend is detected, but video ID cannot be parsed.
    """
    pass


class YoutubeURL(SourceURL):
    """
    for YouTube URLs.
    """

    @classmethod
    def SourceLocationType(cls):
        return 'youtube';

    # Compiled regex (:py:func:`re.compile`) to search code in URL.
    # Example: ``re.compile(r'myvideo\.com/\?code=(?P<code>\w+)')``
    re_code = None

    # Compilede regec (:py:func:`re.compile`) to detect, if input URL is valid for current backend.
    # Example: ``re.compile(r'^http://myvideo\.com/.*')``
    re_detect = None

    # Pattern in which the code is inserted.
    # Example: ``http://myvideo.com?code=%s``
    # :type: str
    pattern_url = None

    pattern_thumbnail_url = None

    re_detect = re.compile(
        r'^(http(s)?://)?(www\.|m\.)?youtu(\.?)be(\.com)?/.*', re.I
    )

    re_code = re.compile(
        r'''youtu(\.?)be(\.com)?/  # match youtube's domains
            (\#/)? # for mobile urls
            (embed/)?  # match the embed url syntax
            (v/)?
            (watch\?v=)?  # match the youtube page url
            (ytscreeningroom\?v=)?
            (feeds/api/videos/)?
            (user\S*[^\w\-\s])?
            (?P<code>[\w\-]{11})[a-z0-9;:@?&%=+/\$_.-]*  # match and extract
        ''',
        re.I | re.X
    )

    pattern_url = '{protocol}://www.youtube.com/embed/{code}'
    pattern_thumbnail_url = '{protocol}://img.youtube.com/vi/{code}/{resolution}'

    resolutions = [
        'maxresdefault.jpg',
        'sddefault.jpg',
        'hqdefault.jpg',
        'mqdefault.jpg',
    ]


    def get_url(self):
        """
        Returns URL folded from :py:data:`pattern_url` and parsed code.
        """
        url = self.pattern_url.format(code=self.code, protocol=self.protocol)
        url += '?' + self.query.urlencode() if self.query else ''
        return url

    def get_thumbnail_url(self):
        """
        Returns thumbnail URL folded from :py:data:`pattern_thumbnail_url` and
        parsed code.

        :rtype: str
        """
        return self.pattern_thumbnail_url.format(code=self.code,
                                                 protocol=self.protocol)

    def _getCode(self):

        match = self.re_code.search(self._source_location)
        if match:
            return match.group('code')

        parsed_url = urllib.parse.urlparse(self._source_location)
        parsed_qs = urllib.parse.parse_qs(parsed_url.query)

        if 'v' in parsed_qs:
            code = parsed_qs['v'][0]
        elif 'video_id' in parsed_qs:
            code = parsed_qs['video_id'][0]
        else:
            raise UnknownIdException('Cannot get ID from `{0}`'.format(self._source_location))

        return code

    def get_thumbnail_url(self):
        """
        Returns thumbnail URL folded from :py:data:`pattern_thumbnail_url` and
        parsed code.

        :rtype: str
        """
        for resolution in self.resolutions:
            temp_thumbnail_url = self.pattern_thumbnail_url.format(
                code=self.code, protocol=self.protocol, resolution=resolution)
            if int(requests.head(temp_thumbnail_url).status_code) < 400:
                return temp_thumbnail_url
        return None



class FilePathURL(SourceURL):
    @classmethod
    def SourceLocationType(cls):
        return 'file_path';

    def __init__(self, source_location):
        self.source_location_type = self.SourceLocationType()
        self._source_location = source_location;

    @classmethod
    def is_valid(cls, source_location):
        return os.path.isfile(source_location);

    def _getCode(self):
        name_parts = os.path.splitext(os.path.basename(self._source_location));
        return name_parts[0];



SOURCE_LOCATION_TYPES = (
    YoutubeURL,
    FilePathURL,
)

def ParseSourseLocation(url):
    """

    :param url:
    :return:
    """
    for backend in SOURCE_LOCATION_TYPES:
        if backend.is_valid(url):
            return backend(url)

    raise UnknownBackendException