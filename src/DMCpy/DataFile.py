# SPDX-License-Identifier: MPL-2.0
import h5py as hdf
import datetime
from h5py._hl import attrs
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import pandas as pd
import DMCpy
import os.path
from DMCpy import InteractiveViewer
from DMCpy.TasUBlibDEG import converterToA3A4Z

import warnings

import copy
from DMCpy._tools import KwargChecker, MPLKwargs, roundPower
from DMCpy import Sample
from DMCpy.FileStructure import HDFCounts, HDFCountsBG, HDFTranslation, HDFTranslationAlternatives, HDFTranslationDefault, HDFTranslationFunctions
from DMCpy.FileStructure import HDFInstrumentTranslation, HDFInstrumentTranslationFunctions, extraAttributes, possibleAttributes 
from DMCpy.FileStructure import HDFTypes, HDFUnits, shallowRead


scanTypes = ['Old Data','Powder','A3']

def decode(item):
    """Test and decode item to utf8"""
    if hasattr(item,'__len__'):
        if len(item)>0:
            if hasattr(item[0],'decode'):
                item = item[0].decode('utf8')
            
                    
    return item


@KwargChecker()
def maskFunction(phi,maxAngle=10.0):
    """Mask all phi angles outside plus/minus maxAngle

    Args:

        - phi (array): Numpy array of phi angles to be masked

    Kwargs:

        - maxAngle (float): Mask points greater than this or less than -maxAngle in degrees (default 10)
    """
    return np.abs(phi)>maxAngle

@KwargChecker()
def findCalibration(fileName):
    """Find detector calibration for specified file

    Args:

        - fileName (str): Name of file for which calibration is needed

    Returns:

        - calibration (array): Detector efficiency

        - calibrationName (str): Name of calibration file used

    Raises:

        - fileNotFoundError

    """

    # Extract only actual file name if path is provided    
    fileName = os.path.split(fileName)[-1]

    # Split name in 'dmcyyyynxxxxxx.hdf'
    year,fileNo = [int(x) for x in fileName[3:].replace('.hdf','').split('n')]

    calibrationDict = DMCpy.calibrationDict

    # Calibration files do not cover the wanted year
    if not year in calibrationDict.keys():
        warnings.warn('Calibration files for year {} (extracted from file name "{}") is'.format(year,fileName)+\
            ' not covered in calibration tables. Please update to newest version by invoking "pip install --upgrade DMCpy"')
        calibration = np.ones_like((128,1152))
        calibrationName = 'None'
        return calibration,calibrationName
        #raise FileNotFoundError('Calibration files for year {} (extracted from file name "{}") is'.format(year,fileName)+\
        #    ' not covered in calibration tables. Please update to newest version by invoking "pip install --upgrade DMCpy"')

    yearCalib = calibrationDict[year]
    
    limits = yearCalib['limits']
    
    # Calibration name is index of the last limit below file number
    idx = np.sum([fileNo>limits])-1
    
    idx = np.max([idx,0]) # ensure that idx is not negative
    
    # Calibration is saved in yearCalib with name of calibration file
    calibrationName = yearCalib['names'][idx]
    calibration = yearCalib[calibrationName]
    return calibration,calibrationName

# Custom class designed to perform a lazy q calculation. Usage:
# All calculations are needed: q[None]
# Only a specific slice is needed: q[:10]
class lazyQ(object):
    def __init__(self,rotationMatrix,q_temp):
        self.rotationMatrix = rotationMatrix
        self.q_temp = q_temp
        
    def __getitem__(self,sl=None):
        return np.einsum('jki,k...->ji...',self.rotationMatrix[:,:,sl].reshape(3,3,-1),self.q_temp)




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


@KwargChecker(include=['radius','twoTheta','verticalPosition','twoThetaPosition','forcePowder','sampleOffsetZ']+list(HDFTranslation.keys()))
def loadDataFile(fileLocation=None,fileType='Unknown',unitCell=None,forcePowder=False,**kwargs):
    """Load DMC data file, either powder or single crystal data.
    
    """
    if fileLocation is None:
        return DataFile()

    if isinstance(fileLocation,(DataFile)):
        if fileLocation.fileType.lower() == 'powder':
            return PowderDataFile(fileLocation,unitCell=unitCell)
        elif fileLocation.fileType.lower() == 'singlecrystal':
            return SingleCrystalDataFile(fileLocation,unitCell=unitCell)
        else:
            return DataFile(fileLocation,unitCell=unitCell)
    elif not os.path.exists(fileLocation): # load file from disk
        raise FileNotFoundError('Provided file path "{}" not found.'.format(fileLocation))

    A3 = shallowRead([fileLocation],['A3'])[0]['A3']

    se_r = shallowRead([fileLocation],['se_r'])[0]['se_r']
    
    T = 'Unknown' # type of datafile

    if A3 is None: # there is no A3 values at all
        T = 'powder'
        
    elif (len(A3) == 1 and bool(se_r.any()) is False) or forcePowder:
        T = 'powder'
    else:
        T = 'singlecrystal'

    ## here be genius function to determine type of data
    
    if fileType.lower() == 'powder' or T == 'powder':
        df = PowderDataFile(fileLocation,unitCell=unitCell,forcePowder=forcePowder)
    elif fileType.lower() == 'singlecrystal' or T == 'singlecrystal':
        df = SingleCrystalDataFile(fileLocation,unitCell=unitCell)
    else:
        df = DataFile(fileLocation,unitCell=unitCell)


    repeats = df.countShape[1]
    # Insert standard values if not present in kwargs
    if not 'radius' in kwargs:
        kwargs['radius'] = 0.8

    if not 'verticalPosition' in kwargs:
        kwargs['verticalPosition'] = np.linspace(-0.1,0.1,repeats,endpoint=True)

    if 'sampleOffsetZ' in kwargs:
        temp_sampleOffsetZ = kwargs['sampleOffsetZ']
        del kwargs['sampleOffsetZ']
    else:
        temp_sampleOffsetZ = None

    # Overwrite parameters provided in the kwargs
    for key,item in kwargs.items():
        setattr(df,key,item)
        
    if 'twoThetaPosition' in kwargs:
        if not 'twoTheta' in kwargs:
            df.twoTheta = np.linspace(0,-132,9*128)+df.twoThetaPosition
        else:
            df.twoTheta = kwargs['twoTheta']
    elif 'twoTheta' in kwargs:
        df.twoTheta = kwargs['twoTheta']
    
    if temp_sampleOffsetZ is None:
        df.initializeQ()
    else:
        df.sampleOffsetZ = temp_sampleOffsetZ
    df.loadNormalization()

    year,month,date = [int(x) for x in df.startTime.replace('T',' ').split(' ')[0].split('-')]
    if year == 2022:
        df.mask[0,-2,:] = True

    if year == 2023:
        df.mask[0,-1,:] = True
        df.mask[0,:,-1] = True

    return df



class DataFile(object):
    @KwargChecker()
    def __init__(self, file=None,unitCell=None,forcePowder=False):
        self.fileType = 'DataFile'
        self._twoThetaOffset = 0.0
        self.monochromatorDistance = 2.82 # 
        self._counts = None
        self._background = None

        if not file is None: 
            if isinstance(file,DataFile): # Copy everything from provided file
                # Copy all file settings
                self.updateProperty(file.__dict__)

            elif os.path.exists(file): # load file from disk
                self.loadFile(file,unitCell=unitCell)


            else:
                raise FileNotFoundError('Provided file path "{}" not found.'.format(file))

    @KwargChecker()
    def loadFile(self,filePath,unitCell=None,forcePowder=False):
        if not os.path.exists(filePath):
            raise FileNotFoundError('Provided file path "{}" not found.'.format(filePath))

        self.folder, self.fileName = os.path.split(filePath)

        # setup standard parameters

        self._wavelength = 0.0


        # Open file in reading mode
        with hdf.File(filePath,mode='r') as f:

            self.sample = Sample.Sample(sample=f.get(HDFTranslation['sample']))
            self.countShape = f.get(HDFCounts).shape
            self.hasBackground = not f.get(HDFCountsBG) is None
            # load standard things using the shallow read
            instr = getInstrument(f)

            if not f.get('/entry/reduction') is None: # Data file is a merged/reduced data file
                red = f['/entry/reduction']

                # Complicated way to avoid having to guess the name of the reduction algorithm.....
                self.original_files = np.asarray([name.decode('UTF8') for name in list(red.values())[0].get('rawdata')]) 
                
            for parameter in HDFTranslation.keys():
                if parameter in ['unitCell','sample','unitCell']:
                    continue
                if parameter in HDFTranslationAlternatives:
                    for entry in HDFTranslationAlternatives[parameter]:
                        value = np.array(f.get(entry))
                        if not value.shape == ():
                            break

                elif parameter in HDFTranslation:
                    value = np.array(f.get(HDFTranslation[parameter]))
                    TrF= HDFTranslationFunctions
                elif parameter in HDFInstrumentTranslation:
                    value = np.array(instr.get(HDFInstrumentTranslation[parameter]))
                    TrF= HDFInstrumentTranslationFunctions

                if value.shape == () or value is None:
                    value = HDFTranslationDefault[parameter]

                else:
                    for func,args in TrF[parameter]:
                        value = getattr(value,func)(*args)
                
                setattr(self,parameter,value)
                
        self.countShape = (1,*self.countShape) # Standard shape
        if not unitCell is None:
            self.sample.unitCell = unitCell
    
    def initializeQ(self):
        if len(self.twoTheta.shape) == 2:
            self.twoTheta, z = np.meshgrid(self.twoTheta[0].flatten(),self.verticalPosition,indexing='xy')
        else:
            self.twoTheta, z = np.meshgrid(self.twoTheta.flatten(),self.verticalPosition,indexing='xy')
            
        self.pixelPosition = np.array([-self.radius*np.sin(np.deg2rad(self.twoTheta)),
                                    self.radius*np.cos(np.deg2rad(self.twoTheta)),
                                    -z]).reshape(3,*self.countShape[1:])
        
        
        #self.Monitor = self.monitor
        
        if np.any(np.isclose(self.monitor,0)): # error mode from commissioning
            self.monitor = np.ones(self.countShape[0])
        
        self.alpha = np.rad2deg(np.arctan2(self.pixelPosition[2],self.radius))
        # Above line makes an implicit call to the self.calculateQ method!
        
        self.calculateQ()
        self.generateMask(maskingFunction=None)


    
    def loadNormalization(self):
        # Load calibration
        try:
            if hasattr(self,'original_files'): # We are working with a converted/merged file
                name = self.original_files[0]
            else:
                name = self.fileName
            self.normalization, self.normalizationFile = findCalibration(name)
        except ValueError:
            self.normalizationFile = 'None'

        if self.normalizationFile == 'None':
            self.normalization = np.ones(self.countShape,dtype=float)
        else:
            
            if self.fileType.lower() == "singlecrystal": # A3 scan
                self.normalization = self.normalization#np.repeat(self.normalization[np.newaxis],self.countShape[0],axis=0)
                #self.normalization.shape = self.countShape
                #self.normalization = self.normalization.reshape(self.countShape)
            else:
                self.normalization = self.normalization.reshape(self.countShape)

    def __len__(self):
        if not hasattr(self,'countShape'):
            if hasattr(self,'counts'):
                self.countShape = self.counts.shape[0]
            else:
                return 0
        return self.countShape[0]

    @property
    def A3(self):
        return self.sample.rotation_angle

    @A3.getter
    def A3(self):
        if not hasattr(self.sample,'rotation_angle'):
            self.sample.rotation_angle = np.array([0.0]*len(self.monitor))
        return self.sample.rotation_angle
        

    @A3.setter
    def A3(self,A3):
        if A3 is None:
            self.sample.rotation_angle = np.array([0.0]*len(self.monitor))
        else:
            self.sample.rotation_angle = A3
        if hasattr(self,'ki'):
            self.calculateQ()
    

    @property
    def twoThetaPosition(self):
        return self._detector_position

    @twoThetaPosition.getter
    def twoThetaPosition(self):
        if not hasattr(self,'_detector_position'):
            self._detector_position = np.array([0.0])
        return self._detector_position+self.twoThetaOffset

    @twoThetaPosition.setter
    def twoThetaPosition(self,twoTheta):
        if twoTheta is None:
            self._detector_position = np.array([0.0]*len(self.A3))
        elif np.isnan(twoTheta):
            self._detector_position = np.array([0.0]*len(self.A3))
        else:
            self._detector_position = np.asarray(twoTheta)
        self.twoTheta = np.repeat((np.linspace(0,-132,1152) + self._detector_position + self._twoThetaOffset)[np.newaxis],self.countShape[1],axis=0)
        if hasattr(self,'_Ki') and hasattr(self,'twoTheta'):
            self.calculateQ()

    

    @property
    def Ki(self):
        return self._Ki

    @Ki.getter
    def Ki(self):
        return self._Ki

    @Ki.setter
    def Ki(self,Ki):
        self._Ki = Ki
        self.wavelength = np.full_like(self.wavelength,2*np.pi/Ki)
        self.calculateQ()

    @property
    def twoThetaOffset(self):
        return self._twoThetaOffset

    @twoThetaOffset.getter
    def twoThetaOffset(self):
        return self._twoThetaOffset

    @twoThetaOffset.setter
    def twoThetaOffset(self,dTheta):
        self._twoThetaOffset = dTheta
        self.twoTheta = np.repeat((np.linspace(0,-132,1152) + self._detector_position + self._twoThetaOffset)[np.newaxis],self.countShape[1],axis=0)
        self.calculateQ()

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.getter
    def wavelength(self):
        if not hasattr(self,'_wavelength'):
            self._wavelength = 0.0
        return self._wavelength

    @wavelength.setter
    def wavelength(self,wavelength):
        self._wavelength = wavelength
        self._Ki = 2*np.pi/wavelength
        self.calculateQ()
    
    @property
    def sampleOffsetZ(self):
        return self._sampleOffsetZ

    @sampleOffsetZ.getter
    def sampleOffsetZ(self):
        if not hasattr(self,'_sampleOffsetZ'):
            self._sampleOffsetZ =  0.0
        return self._sampleOffsetZ

    @sampleOffsetZ.setter
    def sampleOffsetZ(self,sampleOffsetZ):
        self._sampleOffsetZ = sampleOffsetZ
        self.initializeQ()
        self.calculateQ()
        
    def calculateQ(self):
        """Calculate Q and qx,qy,qz using the current A3 values"""
        if not (hasattr(self,'Ki') and hasattr(self,'twoTheta')
                and hasattr(self,'alpha') and hasattr(self,'A3')):
            return 

        self.neu = np.rad2deg(np.arctan2(self.sampleOffsetZ,self.monochromatorDistance))
        self.ki = np.array([0.0,np.cos(np.deg2rad(self.neu)),np.sin(np.deg2rad(self.neu))])*self.Ki# along ki=2pi/lambda with y (Lumsden2005)
        self.ki.shape = (3,1,1)

        self.kf = self.Ki*self.pixelPosition/np.linalg.norm(self.pixelPosition,axis=0)
           
        if self.fileType.lower() == 'singlecrystal': # A3 Scan
            # rotate kf to correct for A3
            zero = np.zeros_like(self.A3)
            ones = np.ones_like(self.A3)
            self.rotMat = np.array([[np.cos(np.deg2rad(self.A3)),np.sin(np.deg2rad(self.A3)),zero],[-np.sin(np.deg2rad(self.A3)),np.cos(np.deg2rad(self.A3)),zero],[zero,zero,ones]])
            self.q_temp = self.kf-self.ki

            self.q = lazyQ(self.rotMat, self.q_temp)

            self.Q = np.repeat(np.linalg.norm(self.q[0],axis=0),self.countShape[0],axis=0)
        else:
            self.qLocal = self.ki-self.kf
            self.Q = np.array([np.linalg.norm(self.qLocal,axis=0)])

        #self.correctedTwoTheta = 2.0*np.rad2deg(np.arcsin(self.wavelength*self.Q[0]/(4*np.pi)))[np.newaxis].repeat(self.Q.shape[0],axis=0)
        

    def generateMask(self,maskingFunction = maskFunction, replace=True, **pars):
        """Generate mask to applied to data in data file
        
        Kwargs:

            - maskingFunction (function): Function called on self.phi to generate mask (default maskFunction)

            - replace (bool): If true new mask replaces old one, otherwise add together (default True)

        All other arguments are passed to the masking function.

        """

        # check if counts attribute is available

        if len(self.countShape) <1:#not hasattr(self,'counts'):
            raise RuntimeError('DataFile does not contain any counts. Look for self.counts but found nothing.')

        if maskingFunction is None:
            if replace:
                self.mask = np.zeros(self.countShape,dtype=bool)
            else:
                self.mask += np.zeros(self.countShape,dtype=bool)
        else:
            if replace:
                self.mask = maskingFunction(self.phi,**pars).reshape(*self.countShape)
            else:
                self.mask += maskingFunction(self.phi,**pars).reshape(*self.countShape)
        
        

    def updateProperty(self,dictionary):
        """Update self with key and values from provided dictionary. Overwrites any properties already present."""
        if isinstance(dictionary,dict):
            for key,item in dictionary.items():
                if key == 'exclude': continue
                if key == 'kwargs': # copy kwargs directly and continue
                    self.kwargs = item
                    continue
                item = decode(item)
                self.__setattr__(key,copy.deepcopy(item))
        else:
            raise AttributeError('Provided argument is not of type dictionary. Received instance of type {}'.format(type(dictionary)))


    @KwargChecker(function=plt.errorbar,include=MPLKwargs)
    def plotDetector(self,ax=None,applyCalibration=True,**kwargs):
        """Plot intensity as function of twoTheta (and vertical position of pixel in 2D)

        Kwargs:

            - ax (axis): Matplotlib axis into which data is plotted (default None - generates new)

            - applyCalibration (bool): If true, take detector efficiency into account (default True)

            - All other key word arguments are passed on to plotting routine

        Returns:
            - ax (matplotlib axis): axis into which the detector is potted
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        
        if applyCalibration:
            intensity=self.intensity/self.monitor.reshape(-1,1,1)#1.0/self.normalization

        count_err = np.sqrt(self.counts)
        intensity_err = count_err/self.monitor.reshape(-1,1,1)
        if applyCalibration:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                intensity_err*=1.0/self.normalization
 


        # If data is one dimensional
        if self.twoTheta.shape[1] == 1:
            if not 'fmt' in kwargs:
                kwargs['fmt'] = '_'

            ax._err = ax.errorbar(self.twoTheta[np.logical_not(self.mask)],intensity[np.logical_not(self.mask)],intensity_err[np.logical_not(self.mask)],**kwargs)
            ax.set_xlabel(r'$2\theta$ [deg]')
            ax.set_ylabel(r'Counts/mon [arb]')

            def format_coord(ax,xdata,ydata):
                if not hasattr(ax,'xfmt'):
                    ax.mean_x_power = roundPower(np.mean(np.diff(ax._err.get_children()[0].get_data()[0])))
                    ax.xfmt = r'$2\theta$ = {:3.'+str(ax.mean_x_power)+'f} Deg'
                if not hasattr(ax,'yfmt'):
                    ymin,ymax,ystep = [f(ax._err.get_children()[0].get_data()[1]) for f in [np.min,np.max,len]]
                    
                    ax.mean_y_power = roundPower((ymax-ymin)/ystep)
                    ax.yfmt = r'Int = {:.'+str(ax.mean_y_power)+'f} cts'

                return ', '.join([ax.xfmt.format(xdata),ax.yfmt.format(ydata)])

            ax.format_coord = lambda format_xdata,format_ydata:format_coord(ax,format_xdata,format_ydata)
        else: # plot a 2D image with twoTheta vs z
            # Set all masked out points to Nan
            intensity[self.mask] = np.nan

            if 'colorbar' in kwargs:
                colorbar = kwargs['colorbar']
                del kwargs['colorbar']
            else:
                colorbar = False
            
            ax._pcolormesh = ax.pcolormesh(np.abs(self.twoTheta),self.pixelPosition[2],np.sum(intensity,axis=0),shading='auto')

            if colorbar:
                ax._col = fig.colorbar(ax._pcolormesh)
                ax._col.set_label('Intensity [cts/Monitor]')
                

            ax.set_xlabel(r'$2\theta$ [deg]')
            ax.set_ylabel(r'z [m]')

        return ax

    @KwargChecker()
    def save(self,filePath,compression=6):
        """Save data file in hdf format.
        
        Args:

            - filePath (path): Path into which file is to be saved

        Kwargs:

            - compression (int): Compression level used by gzip

        """
        if os.path.exists(filePath):
            raise AttributeError('File already exists! ({})'.format(filePath))

        
        with hdf.File(filePath,'w') as f:
    
            # Create correct header info
            f.attrs['NeXus_Version'] = np.string_('4.4.0')
            f.attrs['file_name'] = np.string_(filePath)
            
            
            cT = datetime.datetime.now()
            
            f.attrs['file_time'] = np.string_('{}-{}-{} {}:{}:{}'.format(cT.year,cT.month,cT.day,cT.hour,cT.minute,cT.second))
            f.attrs['instrument'] = np.string_('DMC')
            f.attrs['owner'] = np.string_('Lukas Keller <lukas.keller@psi.ch>')

            entry = f.create_group('entry')
            entry.attrs['NX_class'] = np.string_('NXentry')
            entry.attrs['default'] = np.string_('data')

            # Generate file structure
            DMC = entry.create_group('DMC')
            DMC.attrs['NX_class'] = np.string_('NXinstrument')
            
            SINQ = DMC.create_group('SINQ')
            SINQ.attrs['NX_class'] = np.string_('NXsource')
            SINQ.attrs['name'] = np.string_('SINQ')
            SINQ.attrs['type'] = np.string_('Continuous flux spallation source')
            
            detector = DMC.create_group('detector')
            detector.attrs['NX_class'] = np.string_('NXdetector')

            if self.fileType.lower() != 'singlecrystal':
                position = detector.create_dataset('detector_position',data=np.array(self.twoThetaPosition))
            else:
                position = detector.create_dataset('detector_position',data=np.full(len(self),self.twoThetaPosition))
            
            position.attrs['units'] = np.string_('degree')

            summedCounts = detector.create_dataset('summed_counts',data=self.counts.sum(axis=0))
            summedCounts.attrs['units'] = np.string_('counts')
            
            
            # Generate structure of file
            
            
            
            mono = DMC.create_group('monochromator')
            mono.attrs['NX_class'] = np.string_('NXmonochromator')
            mono.attrs['type'] = np.string_('Pyrolytic Graphite')
            
            wavelength = mono.create_dataset('wavelength',data=np.array([self.wavelength]))
            wavelength.attrs['units'] = 'A'
            
            # data
            data = entry.create_group('data')
            data.attrs['NX_class'] = np.string_('NXdata')
            data.attrs['signal'] = np.string_('data')
            
            Monitor = entry.create_group('monitor')
            Monitor.attrs['NX_class'] = np.string_('NXmonitor')
            
            
            user = entry.create_group('user')
            user.attrs['NX_class'] = np.string_('NXuser')
            
            
            for key,value in HDFTranslation.items():
                if key in ['counts','summedCounts','wavelength','detector_position','twoThetaPosition']: continue
                if 'sample' in value: continue
                selfValue = HDFTypes[key](getattr(self,key))
                
                newEntry = f.create_dataset(value,data=selfValue)
                if key in HDFUnits:
                    newEntry.attrs['units'] = np.string_(HDFUnits[key])

                    
            
            sample = entry.create_group('sample')
            sample.attrs['NX_class'] = np.string_('NXsample')
            
            a3 = sample.create_dataset('rotation_angle',data=self.A3)
            a3.attrs['units'] = 'degree'
            
            self.sample.saveToHdf(sample)
            
            
            
            # data
            data = entry['data']
            data.attrs['NX_class'] = np.string_('NXdata')
            data.attrs['signal'] = np.string_('data')
            
            if self.fileType.lower() != 'singlecrystal':
                Data = data.create_dataset('data',data=self.counts[0],compression=compression)
            else:
                Data = data.create_dataset('data',data=self.counts,compression=compression)
            Data.attrs['units'] = np.string_('A')
            
            # Create link to data in the right place
            data = detector['data'] = Data
            data.attrs['signal'] = np.int32(1)
            data.attrs['target'] = np.string_('/entry/DMC/detector/data')

            
            entry['monitor/monitor'].attrs['units'] = np.string_('counts')


    def __eq__(self,other):
        return len(self.difference(other))==0
    
    def difference(self,other,keys = set(['sample.name','wavelength','counts','A3','twoTheta','fileType','monitor'])):
        """Return the difference between two data files by keys"""
        dif = []
        if not set(self.__dict__.keys()) == set(other.__dict__.keys()): # Check if same generation and type (hdf or nxs)
            return list(set(self.__dict__.keys())-set(other.__dict__.keys()))

        comparisonKeys = keys
        for key in comparisonKeys:
            skey = self
            okey = other
            while '.' in key:
                baseKey,*keys = key.split('.')
                skey = getattr(skey,baseKey)
                okey = getattr(okey,baseKey)
                key = '.'.join(keys)
            if isinstance(skey,np.ndarray):
                try:
                    if not np.all(np.isclose(skey,okey)):
                        if not np.all(np.isnan(skey),np.isnan(okey)):
                            dif.append(key)
                except (TypeError, AttributeError,ValueError):
                    if np.all(skey!=okey):
                        dif.append(key)
            elif not np.all(getattr(skey,key)==getattr(okey,key)):
                dif.append(key)
        return dif


    @property
    def counts(self):
        if self._counts is None:
            if self.hasBackground:
                bg = self.background
            else:
                bg = 0
            with hdf.File(os.path.join(self.folder,self.fileName),mode='r') as f:
                return (np.array(f.get(HDFCounts))).reshape(self.countShape)-bg
        else:
            return self._counts.reshape(self.countShape)
    
    def countsSliced(self,sl):
        if self._counts is None:
            if self.hasBackground:
                bg = self.background[sl]
            else:
                bg = 0
            with hdf.File(os.path.join(self.folder,self.fileName),mode='r') as f:
                return np.array(f.get(HDFCounts)[sl])-bg
        else:
            return self._counts[sl]
        
    @property
    def background(self):
        if self._background is None:
            with hdf.File(os.path.join(self.folder,self.fileName),mode='r') as f:
                if self.backgroundType == 'powder':
                    bg = np.repeat(np.array(f.get(HDFCountsBG))[np.newaxis],repeats=self.countShape[0],axis=0)
                else:
                    if self.fileType == 'powder':
                        bg = np.array(f.get(HDFCountsBG)).sum(axis=(0,1)).reshape(self.countShape)
                    else:
                        bg = np.array(f.get(HDFCountsBG)).reshape(self.countShape)
                return bg
        else:
            return self._background.reshape(self.countShape)
    
    def backgroundSliced(self,sl):
        if self._background is None:
            with hdf.File(os.path.join(self.folder,self.fileName),mode='r') as f:
                if self.backgroundType == 'powder':
                    bg = np.repeat(np.array(f.get(HDFCountsBG))[np.newaxis],repeats=len(sl),axis=0)
                else:
                    bg = np.array(f.get(HDFCountsBG)[sl])
                return bg
        else:
            return self._background[sl]
    
            

    @property
    def intensity(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.fileType.lower() == 'singlecrystal':
                return np.divide(self.counts,self.normalization[np.newaxis])
            else:
                return np.divide(self.counts,self.normalization)

    
    def intensitySliced(self,sl):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.fileType.lower() == 'singlecrystal':
                return np.divide(self.countsSliced(sl),self.normalization[np.newaxis])
            else:
                return np.divide(self.countsSliced(sl),self.normalization)
            #return np.divide(self.countsSliced(sl),self.normalization[sl])

    

    def InteractiveViewer(self,**kwargs):
        if not self.fileType.lower() in ['singlecrystal','powder'] :
            raise AttributeError('Interactive Viewer can only be used for the new data files. Either for powder or for a single crystal A3 scan')
        return InteractiveViewer.InteractiveViewer(self.intensity,self.twoTheta,self.pixelPosition,self.A3,scanParameter = 'A3',scanValueUnit='deg',colorbar=True,**kwargs)

    @property
    def correctedTwoTheta(self):
        return 2.0*np.rad2deg(np.arcsin(self.wavelength*self.Q[0]/(4*np.pi)))[np.newaxis].repeat(self.Q.shape[0],axis=0)

    

    @property
    def phi(self):
        if self.fileType.lower() == 'singlecrystal':
            return np.rad2deg(np.arctan2(self.q[None][2],np.linalg.norm(self.q[None][:2],axis=0)))
        else:
            return np.rad2deg(np.arctan2(self.qLocal[2],np.linalg.norm(self.qLocal[:2],axis=0)))
        
    def setProjectionVectors(self,p1,p2,p3=None):
        """Set or update the projection vectors used for the View3D
        
        Args:

            - p1 (list): New primary projection, in HKL

            - p2 (list): New secondary projection, in HKL

        Kwargs:

            - p3 (list): New tertiary projection, in HKL. If None, orthogonal to p1 and p2 (default None)
        """
        self.sample.setProjectionVectors(p1=p1,p2=p2,p3=p3)


class SingleCrystalDataFile(DataFile):
    def __init__(self,fileType,*args,**kwargs):
        super(SingleCrystalDataFile,self).__init__(fileType,*args,**kwargs)
        self.fileType = 'SingleCrystal'
        self.countShape = (self.countShape[0]*self.countShape[1],128,1152)

    def calcualteHKLToA3A4Z(self,H,K,L,Print=True,A4Sign=-1):
        Qx,Qy,Qz = self.sample.calculateHKLToQxQyQz(H,K,L)
        if Print:
            A3, A4, z = converterToA3A4Z(Qx,Qy,Qz,Ki=self.Ki,Kf=self.Ki,A4Sign=A4Sign,radius=self.radius)
            print(f'Calculated angles for ({H},{K},{L}): \nA3: {np.round(A3,3)} \nA4: {np.round(A4,3)} \nz: {np.round(z,5)}\n')
            print('Disclaimer: You might want to use a peak 180 or 360 deg away.\n')
            if A3 < -180:
                print(f'Alternative A3: {np.round(A3+180,3)} or {np.round(A3+360,3)}')
            if -180 < A3 < 0:
                print(f'Alternative A3: {np.round(A3+180,3)}')
            if 0 < A3 < 180:
                print(f'Alternative A3: {np.round(A3+180,3)} or {np.round(A3-180,3)}')
        else:
            return converterToA3A4Z(Qx,Qy,Qz,Ki=self.Ki,Kf=self.Ki,A4Sign=A4Sign,radius=self.radius)

    

class PowderDataFile(DataFile):
    def __init__(self,fileType,*args,**kwargs):
        super(PowderDataFile,self).__init__(fileType,*args,**kwargs)
        self.fileType = 'Powder'

        if 'forcePowder' in kwargs:
            if kwargs['forcePowder']:
                self._counts = np.sum(self.counts,axis=(0,1))
                self.countShape = (1,128,1152)
                self.monitor = np.array([np.sum(self.monitor)])
        self.counts.shape = (1,128,1152)

