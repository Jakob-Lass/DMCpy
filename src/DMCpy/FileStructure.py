import numpy as np
from collections import defaultdict
import warnings, os
import h5py as hdf

HDFCounts = 'entry1/area_detector2/data'
HDFCountsBG = 'entry/data/background'
## Dictionary for holding hdf position of attributes. HDFTranslation['a3'] gives hdf position of 'a3'
HDFTranslation = {'sample':'/entry1/sample',
                  'sampleName':'/entry1/sample/name',
                  'monitor':None,#'entry1/monitor/monitor',
                  'monitor1':'entry1/control/Monitor',
                  'unitCell':'/entry1/sample/cell',
                  'UB':'entry1/sample/UB',
                  #'counts':'entry1/DMC/detector/data',
                  #'background':'entry1/DMC/detector/background',
                  
                  'radius':'entry1/ZEBRA/area_detector2/distance',
                  'wavelength':'entry1/ZEBRA/monochromator/wavelength',
                  'twoThetaPosition':'entry1/ZEBRA/area_detector2/polar_angle',
                  'nu':'entry1/ZEBRA/area_detector2/tilt_angle',
                  'startTime':'entry1/start_time',
                  'time':'Henning',# Is to be caught by HDFTranslationAlternatives 'entry1/monitor/time',
                  'endTime':'entry1/end_time',
                  'comment':'entry1/comment',
                  'proposal':'entry1/proposal_id',
                  'proposalTitle':'entry1/proposal_title',
                  'localContact':'entry1/local_contact/name',
                  'proposalUser':'entry1/proposal_user/name',
                  'proposalEmail':'entry1/proposal_user/email',
                  'user':'entry1/user/name',
                  'email':'entry1/user/email',
                  'address':'entry1/user/address',
                  'affiliation':'entry1/user/affiliation',
                  'A3':'entry1/sample/rotation_angle',
                  'phiRaw':'entry1/sample/phi',
                  'chi':'entry1/sample/chi',
                  'se_r':'entry/sample/se_r',
                  'temperature':'entry1/sample/temperature',
                  'magneticField':'entry1/sample/magnetic_field',
                  'electricField':'entry1/sample/electric_field',
                  'title':'entry1/title',
                  'absoluteTime':'entry1/control/time',
                  'protonBeam':None,# 'entry1/proton_beam/data'
                  'instrGeometry':'entry1/zebra_mode'
}

HDFTranslationAlternatives = { # Alternatives to the above list. NOTTICE: The above positions are not checked if an entry in HDFTranslationAlternatives is present
    'time':['entry1/control/time','entry1/monitor/monitor'],
    'monitor':['entry1/control/Monitor','entry1/control/data'],
    'protonBeam':['entry1/proton_beam/data','entry1/monitor/proton_charge'],
    'A3':['entry1/sample/rotation_angle','entry1/area_detector2/rotation_angle'],
    'temperature':['entry1/sample/temperature','entry1/sample/Ts/value']
}
## Dictionary for holding standard values 

HDFTranslationDefault = {'twoThetaPosition':np.array([0.0]),
                         'comment': 'No Comments',
                         'endTime': '20yy-mm-dd hh:mm:ss',
                         'proposalTitle': 'Unknown Title',
                         'localContact': 'Unknown Local Contact',
                         'proposalUser': 'Unknown User',
                         'proposalEmail': 'Unknown Email',
                         'address': 'Unknown Address',
                         'affiliation': 'Unknown Affiliation',
                         'scanCommand': 'Unknown scanCommand',

                         'wavelength_raw':np.array([2.0]),
                         'monitor1':np.array([0.0]),

                         'temperature': np.array([0.0]),
                         'magneticField': np.array([0.0]),
                         'electricField': np.array([0.0]),

                         'absoluteTime': np.array([0.0]),
                         'protonBeam': np.array([0.0]),
                         'se_r': np.array([0.0]),

                         'backgroundType': 'None',
                         'instrGeometry': 'nb'

}

## Default dictionary to perform on loaded data, i.e. take the zeroth element, swap axes, etc

HDFTranslationFunctions = defaultdict(lambda : [])
HDFTranslationFunctions['sampleName'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['mode'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['startTime'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['wavelength'] = [['mean',[]]]
HDFTranslationFunctions['wavelength_raw'] = [['mean',[]]]
HDFTranslationFunctions['twoThetaPosition'] = [['__getitem__',[0]]]
HDFTranslationFunctions['endTime'] = [['__getitem__',[0]]]
HDFTranslationFunctions['experimentalIdentifier'] = [['__getitem__',[0]]]
HDFTranslationFunctions['comment'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['proposal'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['proposalTitle'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['localContact'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['proposalUser'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['proposalEmail'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['user'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['email'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['instrGeometry'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['address'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['affiliation'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['scanCommand'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['title'] = [['__getitem__',[0]],['decode',['utf8']]]
HDFTranslationFunctions['backgroundType'] = [['__getitem__',[0]],['decode',['utf8']]]



HDFInstrumentTranslation = {
}

HDFInstrumentTranslationFunctions = defaultdict(lambda : [])
# HDFInstrumentTranslationFunctions['counts'] = [['swapaxes',[1,2]]]
HDFInstrumentTranslationFunctions['twoThetaPosition'] = [['mean',]]
HDFInstrumentTranslationFunctions['wavelength'] = [['mean',]]
HDFInstrumentTranslationFunctions['wavelength_raw'] = [['mean',]]

extraAttributes = ['name','fileLocation']

def possibleAttributes(instr=None):
    # instr argument kept for compatibility; ZEBRA-only underlying translation used
    pA = list(HDFTranslation.keys())+list(HDFInstrumentTranslation.keys())+extraAttributes
    pA.sort(key=lambda v: v.lower())
    return pA


HDFTypes = defaultdict(lambda: lambda x: np.array([np.string_(x)]))
HDFTypes['monitor'] = np.array
HDFTypes['monitor1'] = np.array
HDFTypes['monochromatorCurvature'] = np.array
HDFTypes['monochromatorVerticalCurvature'] = np.array
HDFTypes['monochromatorGoniometerLower'] = np.array
HDFTypes['monochromatorGoniometerUpper'] = np.array
HDFTypes['monochromatorRotationAngle'] = np.array
HDFTypes['monochromatorTakeoffAngle'] = np.array
HDFTypes['monochromatorTranslationLower'] = np.array
HDFTypes['monochromatorTranslationUpper'] = np.array
HDFTypes['wavelength'] = np.array
HDFTypes['wavelength_raw'] = np.array
HDFTypes['twoThetaPosition'] = np.array
HDFTypes['A3'] = np.array
HDFTypes['chi'] = np.array
HDFTypes['phiRaw'] = np.array
HDFTypes['nu'] = np.array
# HDFTypes['mode'] = lambda x: np.array([np.string_(x)])
HDFTypes['preset'] = np.array
# HDFTypes['startTime'] = np.string_
HDFTypes['time'] = np.array
# HDFTypes['endTime'] = np.string_
# HDFTypes['comment'] = np.string_
HDFTypes['absoluteTime'] = np.array
HDFTypes['protonBeam'] = np.array


HDFUnits = {
    'monitor':'counts',
    'monochromatorCurvature':'degree',
    'monochromatorVerticalCurvature':'degree',
    'monochromatorGoniometerLower':'degree',
    'monochromatorGoniometerUpper':'degree',
    'monochromatorRotationAngle':'degree',
    'monochromatorTakeoffAngle':'degree',
    'monochromatorTranslationLower':'mm',
    'monochromatorTranslationUpper':'mm',
    'twoThetaPosition':'degree',
    'monitor':'counts',
    'monitor1':'counts',
    'protonBeam':'uA',
    'wavelength':'A',
    'wavelength_raw':'A'
}

def getNX_class(x,y,attribute):
    try:
        variableType = y.attrs['NX_class']
    except:
        variableType = ''
    if variableType==attribute:
        return x

def getInstrument(file):
    location = file.visititems(lambda x,y: getNX_class(x,y,b'NXinstrument'))
    return file.get(location)

def shallowRead(files,parameters,instr=None):
    # ZEBRA-only shallow read: instr argument accepted for compatibility
    if instr is None:
        instr = 'ZEBRA'
    parameters = np.array(parameters)
    values = []
    possible = []
    for p in parameters:
        possible.append(p in possibleAttributes(instr))
    
    if not np.all(possible):
        if np.sum(np.logical_not(possible))>1:
            raise AttributeError('Parameters {} not found'.format(parameters[np.logical_not(possible)]))
        else:
            raise AttributeError('Parameter {} not found'.format(parameters[np.logical_not(possible)]))
    
    for file in files:
        vals = {}
        vals['file'] = file
        with hdf.File(file,mode='r') as f:
            for p in parameters:
                if p == 'name':
                    v = os.path.basename(file)
                    vals[p] = v
                    continue
                elif p == 'fileLocation':
                    v = os.path.dirname(file)
                    vals[p] = v
                    continue
                elif p in HDFTranslationAlternatives:
                    for entry in HDFTranslationAlternatives[p]:
                        v = np.array(f.get(entry))
                        if not v.shape == ():
                            TrF= HDFTranslationFunctions
                            break


                elif p in HDFTranslation:
                    v = np.array(f.get(HDFTranslation[p]))
                    TrF= HDFTranslationFunctions
                elif p in HDFInstrumentTranslation:
                    v = np.array(f.get(HDFInstrumentTranslation[p]))
                    TrF= HDFInstrumentTranslationFunctions
                else:
                    raise AttributeError('Parameter "{}" not found'.format(p))
                for func,args in TrF[p]:
                    try:
                        v = getattr(v,func)(*args)
                    except (IndexError,AttributeError):
                        warnings.warn('Parameter "{}" not found in file "{}"'.format(p,file))
                        v = None
                        
                vals[p] = v
        values.append(vals)

    return values
