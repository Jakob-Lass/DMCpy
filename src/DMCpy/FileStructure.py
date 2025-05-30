# SPDX-License-Identifier: MPL-2.0
import numpy as np
from collections import defaultdict
import warnings, os
import h5py as hdf


HDFCounts = 'entry/DMC/detector/data'
HDFCountsBG = 'entry/data/background'
## Dictionary for holding hdf position of attributes. HDFTranslation['a3'] gives hdf position of 'a3'
HDFTranslation = {'sample':'/entry/sample',
                  'sampleName':'/entry/sample/name',
                  'monitor':None,#'entry/monitor/monitor',
                  'monitor1':'entry/monitor/monitor1',
                  'unitCell':'/entry/sample/unit_cell',
                  #'counts':'entry/DMC/detector/data',
                  #'background':'entry/DMC/detector/background',
                  'backgroundType':'entry/data/backgroundType',
                  'summedCounts': 'entry/DMC/detector/summed_counts',
                  'monochromatorCurvature':'entry/DMC/monochromator/curvature',
                  'monochromatorVerticalCurvature':'entry/DMC/monochromator/curvature_vertical',
                  'monochromatorGoniometerLower':'entry/DMC/monochromator/goniometer_lower',
                  'monochromatorGoniometerUpper':'entry/DMC/monochromator/goniometer_upper',
                  'monochromatorRotationAngle':'entry/DMC/monochromator/rotation_angle',
                  'monochromatorTakeoffAngle':'entry/DMC/monochromator/takeoff_angle',
                  'monochromatorTranslationLower':'entry/DMC/monochromator/translation_lower',
                  'monochromatorTranslationUpper':'entry/DMC/monochromator/translation_upper',
                  

                  'wavelength':'entry/DMC/monochromator/wavelength',
                  'wavelength_raw':'entry/DMC/monochromator/wavelength_raw',
                  'twoThetaPosition':'entry/DMC/detector/detector_position',
                  'mode':'entry/monitor/mode',
                  'preset':'entry/monitor/preset',
                  'startTime':'entry/start_time',
                  'time':'Henning',# Is to be caught by HDFTranslationAlternatives 'entry/monitor/time',
                  'endTime':'entry/end_time',
                  'comment':'entry/comment',
                  'proposal':'entry/proposal_id',
                  'proposalTitle':'entry/proposal_title',
                  'localContact':'entry/local_contact/name',
                  'proposalUser':'entry/proposal_user/name',
                  'proposalEmail':'entry/proposal_user/email',
                  'user':'entry/user/name',
                  'email':'entry/user/email',
                  'address':'entry/user/address',
                  'affiliation':'entry/user/affiliation',
                  'A3':'entry/sample/rotation_angle',
                  'se_r':'entry/sample/se_r',
                  'temperature':'entry/sample/temperature',
                  'magneticField':'entry/sample/magnetic_field',
                  'electricField':'entry/sample/electric_field',
                  'scanCommand':'entry/scancommand',
                  'title':'entry/title',
                  'absoluteTime':'entry/control/absolute_time',
                  'protonBeam':None# 'entry/proton_beam/data'
}

HDFTranslationAlternatives = { # Alternatives to the above list. NOTTICE: The above positions are not checked if an entry in HDFTranslationAlternatives is present
    'time':['entry/monitor/time','entry/monitor/monitor'],
    'monitor':['entry/monitor/monitor','entry/monitor/monitor2'],
    'protonBeam':['entry/proton_beam/data','entry/monitor/proton_charge']
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

                         'backgroundType': 'None'
                         
                         

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

possibleAttributes = list(HDFTranslation.keys())+list(HDFInstrumentTranslation.keys())+extraAttributes
possibleAttributes.sort(key=lambda v: v.lower())

HDFTypes = defaultdict(lambda: lambda x: np.array([np.bytes_(x)]))
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
# HDFTypes['mode'] = lambda x: np.array([np.bytes_(x)])
HDFTypes['preset'] = np.array
# HDFTypes['startTime'] = np.bytes_
HDFTypes['time'] = np.array
# HDFTypes['endTime'] = np.bytes_
# HDFTypes['comment'] = np.bytes_
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

def shallowRead(files,parameters):

    parameters = np.array(parameters)
    values = []
    possibleAttributes.sort(key=lambda v: v.lower())
    possible = []
    for p in parameters:
        possible.append(p in possibleAttributes)
    
    if not np.all(possible):
        if np.sum(np.logical_not(possible))>1:
            raise AttributeError('Parameters {} not found'.format(parameters[np.logical_not(possible)]))
        else:
            raise AttributeError('Parameter {} not found'.format(parameters[np.logical_not(possible)]))
    
    for file in files:
        vals = {}
        vals['file'] = file
        with hdf.File(file,mode='r') as f:
            instr = getInstrument(f)
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
                    v = np.array(instr.get(HDFInstrumentTranslation[p]))
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