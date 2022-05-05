from .AParamDict import *
import pickle as pickle
import os
class AFuncDict(AParamDict):
    """AFuncDict (class): Extends AParamDict so that functions can be assigned to features and called whenever
        computing those features is necessary.
        Attributes:
            data: name -> value, params
            feature funcs: name -> function for evaluating
    """

    def __init__(self, owner=None, name=None, path=None):
        AParamDict.__init__(self, owner=owner, name=name, path=path);
        self.functions = {};

    def getEntry(self, name=None, params=None, force_recompute=False):
        d = self.data.get(name);
        if((d is not None) and (not force_recompute)):
            return d;
        else:
            f = self.getFunction(name=name);
            if(f is not None):
                if(params is not None):
                    if(not params.get('force_recompute')):
                        params.update(dict(force_recompute=force_recompute));
                    self.setValue(name=name, value=f(self=self.owner, **params), params=params, modified=True);
                else:
                    self.setValue(name=name, value=f(self=self.owner, force_recompute=force_recompute), params=params, modified=True);
            return self.data.get(name);
        return None;

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

    def getFunction(self, name=None):
        return self.functions.get(name);

    def setValue(self, name, value=None, params=None, modified=True):
        self.data[name]['value']=value;
        self.data[name]['params']=params;
        self.setEntryModified(name=name, is_modified=modified)
        #self.data[name]['modified']=modified;

    def setFunction(self, name, function=None):
        self.functions[name]=function;

    def saveEntry(self, name, path, force=False):
        """Save one entry to one file."""
        if(self.data.get(name) is None):
            return None;
        if(self.isEntryModified(name=name) or force or (not os.path.isfile(path))):
            #pickleToPath
            f = open(path, 'wb');
            pickle.dump(self.getEntry(name=name), f, protocol=2);
            f.close();
            self.setEntryModified(name=name, is_modified=False);
            #assert(False), "should not be saving in this test";
        return True;

    def setEntryModified(self, name, is_modified=True):
        self.data[name]['modified']=is_modified;
        if(is_modified):
            self.setModified(is_modified=True);

    def isEntryModified(self, name):
        entry = self.data.get(name);
        if(entry is not None):
            m=entry.get('modified');
            if(m is not None):
                return m;
            else:
                return True;
        else:
            assert(False), "checking mod bit on entry that does not exist"

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
            #assert(False), "should not be saving in this test";
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

    def getFunctionList(self):
        return list(self.functions.keys());
