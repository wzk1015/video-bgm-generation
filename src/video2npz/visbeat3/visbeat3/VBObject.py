import os
import json
from .VisBeatImports import *
from .AFuncDict import *
from .AObject import AObject

class VBObject(AObject):
    """VBObject (class): This is a paarent class used to implement common serialization and other functions. There ends
        up being three different dictionaries of data.
        a_info - for small labels and such. Part of AObject.
        a_data - for data to be computed in experiments. I generally use this for things I don't want to automatically
                save out to file.
        features - these are features tied to the results of functions, and manager classes (e.g. VideoSource) will save
                these to disk for future use.

        FEATURE_FUNCS is a dictionary mapping the names of features to their corresponding functions.
    """
    FEATURE_FUNCS={};

    def __init__(self, path=None):
        AObject.__init__(self, path=path);

    def initializeBlank(self):
        AObject.initializeBlank(self);
        self.a_info.update({'VBObjectType': self.VBOBJECT_TYPE()});
        self.features = AFuncDict(owner=self, name='features');
        self.features.functions.update(self.FEATURE_FUNCS);

    def saveFeature(self, name, path):
        """Subclasses can implement version of this that will check members for features if those features arent found here."""
        return self.features.saveEntry(name=name, path=path);

    def saveFeatures(self, path):
        return self.features.save(path=path);

    def loadFeature(self, name, path):
        """Subclasses can implement version of this that will check whether feature is registered before loading."""
        return self.features.loadEntry(name=name, path=path);

    def loadFeatures(self, path):
        return self.features.load(path=path);

    def getFeature(self,name, force_recompute=False, **kwargs):
        """Understood to get the value of a feature. can automatically recompute if feature has registered function."""
        params = kwargs;
        assert (not kwargs.get('params')), "STILL TRYING TO USE PARAMS INSTEAD OF KWARGS. FIX THIS";
        return self.features.getValue(name=name, params=kwargs, force_recompute=force_recompute);

    def getFeatureEntry(self, name, params=None, force_recompute=False):
        return self.features.getEntry(name=name, params=params, force_recompute=force_recompute);

    def getFeatureParams(self, name):
        return self.features.getParams(name=name);

    def setFeature(self, name, value, params=None):
        rval = self.features.setEntry(name=name, d=dict(value=value, params=params));
        self.features.setEntryModified(name=name, is_modified=True);
        return rval;

    def removeFeature(self, name, assert_if_absent=True, set_modified=True):
        self.features.removeEntry(name=name, assert_if_absent=assert_if_absent, set_modified=set_modified);



    def hasFeature(self, name):
        """Just checks to see if it's there."""
        return self.features.hasEntry(name=name);

    def getFeatureFunction(self, feature_name):
        return self.features.getFunction(name=feature_name);

    def getFeaturesList(self):
        return self.features.getKeyList();

    def getFeatureFunctionsList(self):
        return self.features.getFunctionList();

    def clearFeatureFiles(self, features_to_clear=None, **kwargs):
        if(self.clear_feature_files_func):
            self.clear_feature_files_func(self, features_to_clear=features_to_clear, **kwargs);
        else:
            VBWARN("CLEAR FEATURE FILES FUNCTION HAS NOT BEEN PROVIDED FOR {} INSTANCE".format(self.VBOBJECT_TYPE()));


    ########### VIRTUAL FUNCTIONS #############

    def AOBJECT_TYPE(self):
        return 'VBObject';

    def VBOBJECT_TYPE(self):
        return self.AOBJECT_TYPE();


    # ##Example of how these functions should be written ina  subclass, for 'AssetManager' class
    # def VBOBJECT_TYPE(self):
    #     return 'AssetManager';
    #
    # def toDictionary(self):
    #     d = VBObject.toDictionary(self);
    #     #serialize class specific members
    #     return d;
    #
    # def initFromDictionary(self, d):
    #     VBObject.initFromDictionary(self, d);
    #     #do class specific inits with d;
