from .AImports import *
class AParamDict(object):
    """AParamDict (class): Dictionary that stores values and the parameters used to compute those values. I use this
        class for two things. The first is to store parameters for reproducability. The second is to only recompute
        values when functions are called with different parameters (some of this latter functionality is tied up in
        code that isn't part of the Lite release).

        Attributes:
            data: name -> value, params
    """

    def __init__(self, owner=None, name=None, path=None):
        self.name=name;
        self.data = {};
        self.owner=owner;
        self.modified=False;

    def getEntry(self, name=None, params=None, force_recompute=False):
        d = self.data.get(name);
        if((d is not None) and (not force_recompute)):
            return d;
        return None;

    def setEntry(self, name, d):
        assert(name!='all' and name!='each'),"Entry named '{}' is reserved in AParamDict".format(name);
        self.data[name]=d;

    def removeEntry(self, name, assert_if_absent=True, set_modified = True):
        if(assert_if_absent):
            assert(name in self.data),"Tried to remove entry {} that was not already in {}".format(name, self.__class__);
        popentry = self.data.pop(name, None);
        if(set_modified):
            self.setModified(set_modified);
        return popentry;


    def hasEntry(self, name=None):
        return (self.data.get(name) is not None);

    def getValue(self, name=None, params=None, force_recompute=False):
        d = self.getEntry(name=name, params=params, force_recompute=force_recompute);
        if(d is not None):
            return d.get('value');
        else:
            return None;

    def getParams(self, name=None):
        d = self.data.get(name);
        if(d is not None):
            return d.get('params');
        else:
            return None;

    def setValue(self, name, value=None, params=None, modified=True):
        self.data[name]['value']=value;
        self.data[name]['params']=params;
        self.setEntryModified(name=name, is_modified=modified)

    def saveEntry(self, name, path, force=False):
        """Save one entry to one file."""
        if(self.hasEntry(name=name) is None):
            return None;
        if(self.isEntryModified(name=name) or force or (not os.path.isfile(path))):
            f = open(path, 'wb');
            pickle.dump(self.getEntry(name=name), f, protocol=2);
            f.close();
            self.setEntryModified(name=name, is_modified=False);
        return True;

    def setEntryModified(self, name, is_modified=True):
        self.data[name]['modified']=is_modified;
        if(is_modified):
            self.setModified(is_modified=True);

    def isEntryModified(self, name):
        m=self.data[name].get('modified');
        if(m is not None):
            return m;
        else:
            return True;

    def isModified(self):
        return self.modified;

    def setModified(self, is_modified):
        self.modified=is_modified;

    def save(self, path, force=False):
        """save all entries to one file."""
        if(force or self.isModified()):
            f = open(path, 'wb');
            pickle.dump(self.data, f, protocol=2);
            f.close();
            self.setModified(is_modified=False);
        return True;

    def loadEntry(self, name, path):
        """load one entry from one file."""
        f=open(path, 'rb');
        self.setEntry(name=name, d=pickle.load(f));
        f.close();
        self.setEntryModified(name=name, is_modified=False);
        return True;

    def load(self, path):
        """Load a set of entries all from one file."""
        f=open(path, 'rb');
        newd = pickle.load(f);
        self.data.update(newd);
        f.close();
        return True;

    def getKeyList(self):
        return list(self.data.keys());