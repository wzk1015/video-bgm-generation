import os
import subprocess



def uiGetFilePath(initial_path=None):
    try:
        if(initial_path):
            output = subprocess.check_output("osascript -e 'set strPath to POSIX file \"{}\"' -e 'set theDocument to choose file with prompt \"Please select a document to process:\" default location strPath' -e 'set theDocument to (the POSIX path of theDocument)'".format(initial_path), shell=True)
        else:
            output = subprocess.check_output("osascript -e 'set theDocument to choose file with prompt \"Please select a document to process:\"' -e 'set theDocument to (the POSIX path of theDocument)'", shell=True)
        return output.replace('\n', '');
    except subprocess.CalledProcessError as e:
        print((e.output));
    # assert(False)
    #     grabpath = get_ipython().run_cell_magic(u'bash', u'', "osascript -e 'set theDocument to choose file with prompt \"Please select a document to process:\"' -e 'set theDocument to (the POSIX path of theDocument)'>&2")

def uiGetDirectory(initial_path=None):
    try:
        if(initial_path):
            output = subprocess.check_output("osascript -e 'set strPath to POSIX file \"{}\"' -e 'set thedir to choose folder with prompt \"Please select a file:\" default location strPath' -e 'set thedir to (the POSIX path of thedir)'".format(initial_path), shell=True)
        else:
            output = subprocess.check_output("osascript -e 'set thedir to choose folder with prompt \"Please select a directory:\"' -e 'set thedir to (the POSIX path of thedir)'", shell=True)
        return output.replace('\n', '');
    except subprocess.CalledProcessError as e:
        print((e.output));
    # assert(False)

def uiGetSaveFilePath(initial_path=None, file_extension=None):
    try:
        osastr = "osascript ";
        if(initial_path):
            osastr = osastr+"-e 'set strPath to POSIX file \"{}\"' ".format(initial_path);
        osastr = osastr+"-e 'set theDocument to choose file name with prompt \"Save As File:\" ";
        if(initial_path):
            osastr = osastr+"default location strPath";
        osastr = osastr+"' ";
        osastr = osastr+"-e 'set theDocument to (the POSIX path of theDocument)'"
        output = subprocess.check_output(osastr, shell=True);
        ostring = output.replace('\n', '');
        if (file_extension is not None):
            if (not ostring.endswith(file_extension)):
                ostring = ostring + file_extension;
        return ostring;
    except subprocess.CalledProcessError as e:
        AWARN('ERROR')
        print((e.output));

def showInFinder(path):
    return openOSX(get_dir_from_path(path));

def openOSX(path):
    return subprocess.check_output("open {}".format(put_string_in_quotes(path)), shell=True);

def put_string_in_quotes(s):
    return "\""+s+"\""

def get_file_name_from_path(pth):
    return os.path.split(pth)[1];

def get_dir_from_path(pth):
    return (os.path.split(pth)[0]+os.sep);