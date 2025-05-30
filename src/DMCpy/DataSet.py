# SPDX-License-Identifier: MPL-2.0
import h5py as hdf
import numpy as np
import pickle as pickle
import matplotlib.pyplot as plt
import pandas as pd
import shutil
import os, copy
import json, os, time
from DMCpy import DataFile, _tools, Viewer3D, RLUAxes, TasUBlibDEG
from DMCpy.FileStructure import shallowRead, HDFCountsBG, HDFTranslation
from DMCpy._tools import gauss, gauss_fit
import warnings
import DMCpy

class DataSet(object):
    def __init__(self, dataFiles=None,unitCell=None,forcePowder=False,**kwargs):
        """DataSet object to hold a series of DataFile objects
        Kwargs:
            - dataFiles (list): List of data files to be used in reduction (default None)
        Raises:
            - NotImplementedError
            - AttributeError
        """

        if dataFiles is None:
            self.dataFiles = []
        else:
            if isinstance(dataFiles,(str,DataFile.DataFile)): # If either string or DataFile instance wrap in a list
                dataFiles = [dataFiles]
            try:
                self.dataFiles = [DataFile.loadDataFile(dF,unitCell=unitCell,forcePowder=forcePowder,**kwargs) if isinstance(dF,(str)) else dF for dF in dataFiles]
            except TypeError:
                raise AttributeError('Provided dataFiles attribute is not iterable, filepath, or of type DataFile. Got {}'.format(dataFiles))
            
            self._getData()

    def _getData(self,verbose=False):
        self.type = self[0].fileType
        return

        #data file lengths
        lengths = np.asarray([len(df) for df in self])

        # Collect parameters listed below across data files into self
        for parameter in ['counts','monitor','twoTheta','correctedTwoTheta','fileName','pixelPosition','wavelength','mask','normalization','normalizationFile','time','temperature']:
            if not np.all(lengths==lengths[0]):
                setattr(self,parameter,np.array([getattr(d,parameter) for d in self],dtype=object))
                if verbose: print('file length is not the same for all files => dtype=object')
            else:
                if parameter in ("fileName", "normalizationFile"):
                    dtype = object
                elif parameter in ("mask", ):
                    dtype = bool
                else:
                    dtype = float

                setattr(self,parameter,np.array([getattr(d,parameter) for d in self],dtype=dtype))
                if verbose: print('file length is the same for all files => dtype is induvidual')

        
        types = [df.fileType for df in self]
        if len(types)>1:
            if not np.all([types[0] == t for t in types[1:]]):
                raise AttributeError('Provided data files have different types!\n'+'\n'.join([df.fileName+': '+df.scanType for df in self]))
        self.type = types[0]


    def __len__(self):
        """return number of DataFiles in self"""
        return len(self.dataFiles)
        

    def __eq__(self,other):
        """Check equality to another object. If they are of the same class (DataSet) and have the same attribute keys, the compare equal"""
        return np.logical_and(set(self.__dict__.keys()) == set(other.__dict__.keys()),self.__class__ == other.__class__)


    def __getitem__(self,index):

        try:
            return self.dataFiles[index]
        except IndexError:
            raise IndexError('Provided index {} is out of bounds for DataSet with length {}.'.format(index,len(self)))

    
    def __iter__(self):
        self._index=0
        return self
    
    def __next__(self):
        if self._index >= len(self):
            raise StopIteration
        result = self.dataFiles[self._index]
        self._index += 1
        return result

    def next(self):
        return self.__next__()

    def append(self,item):
        """Append data file to data set."""

        try:
            if isinstance(item,(str,DataFile.DataFile)): # A file path or DataFile has been provided
                item = [item]
            for f in item:
                if isinstance(f,str):
                    f = DataFile.loadDataFile(f)
                self.dataFiles.append(f)
        except Exception as e:
            raise(e)
        self._getData()

    def __delitem__(self,index):
        """Delete data file by index."""
        if index < len(self.dataFiles):
            del self.dataFiles[index]
        else:
            raise IndexError('Provided index {} is out of bounds for DataSet with length {}.'.format(index,len(self.dataFiles)))
        self._getData


    @property
    def sample(self):
        return [df.sample for df in self]

    @sample.getter
    def sample(self):
        return [df.sample for df in self]

    @sample.setter
    def sample(self,sample):
        for df in self:
            df.sample = sample

    def generateMask(self,maskingFunction = DataFile.maskFunction, replace=True, **pars):
        """Generate mask to applied to data in data file
        
        Kwargs:
            - maskingFunction (function): Function called on self.phi to generate mask (default maskFunction)
            - replace (bool): If true new mask replaces old one, otherwise add together (default True)
        All other arguments are passed to the masking function.
        """
        for d in self:
            d.generateMask(maskingFunction,replace=replace,**pars)
        self._getData()

    @_tools.KwargChecker()
    def sumDetector(self,twoThetaBins=None,applyCalibration=True,correctedTwoTheta=True,dTheta=0.125):
        """Find intensity as function of either twoTheta or correctedTwoTheta
        Kwargs:
            - twoThetaBins (list): Bins into which 2theta is to be binned (default min(2theta),max(2theta) in steps of 0.5)
            - applyCalibration (bool): If true, take detector efficiency into account (default True)
            - correctedTwoTheta (bool): If true, use corrected two theta, otherwise sum vertically on detector (default True)
        Returns:
            - twoTheta
            - Normalized Intensity
            - Normalized Intensity Error
            - Total Monitor
        """

        if correctedTwoTheta: 
            twoTheta = np.concatenate([df.correctedTwoTheta[np.logical_not(df.mask)] for df in self],axis=0)
        else:
            twThetaList = []
            for df in self:
                if len(df.twoTheta.shape) == 2: # shape is (df,z,twoTheta), needs to be passed as (df,n,z,twoTheta)
                    twThetaList.append(df.twoTheta[np.newaxis].repeat(df.countShape[0],axis=0)[np.logical_not(df.mask)]) # n = scan steps
                else:
                    twThetaList.append(df.twoTheta[np.logical_not(df.mask)])
            twoTheta = np.concatenate(twThetaList,axis=0)
            
            

        if self.type.lower() == 'powder':
            twoTheta = np.absolute(twoTheta)

        if twoThetaBins is None:
            anglesMin = np.min(twoTheta)
            anglesMax = np.max(twoTheta)
            twoThetaBins = np.arange(anglesMin-0.5*dTheta,anglesMax+0.51*dTheta,dTheta)

        if self.type.lower() == 'singlecrystal':
            monitorRepeated = np.array([np.ones(df.countShape)*df.monitor.reshape(-1,1,1) for df in self])
        else:
            monitorRepeated = np.concatenate([np.repeat(np.repeat(df.monitor[:,np.newaxis,np.newaxis],df.countShape[-2],axis=1),df.countShape[-1],axis=2)[np.logical_not(df.mask)] for df in self])
            
        counts = np.concatenate([df.counts[np.logical_not(df.mask)] for df in self])
        
        summedRawIntensity, _ = np.histogram(twoTheta,bins=twoThetaBins,weights=counts)

        if applyCalibration:
            normalization = np.concatenate([np.repeat(df.normalization,df.countShape[0],axis=0)[np.logical_not(df.mask)] for df in self])
            summedMonitor, _ = np.histogram(twoTheta,bins=twoThetaBins,weights=monitorRepeated*normalization)
        else:
            summedMonitor, _ = np.histogram(twoTheta,bins=twoThetaBins,weights=monitorRepeated)
        
        #inserted, _  = np.histogram(twoTheta[np.logical_not(self.mask)],bins=twoThetaBins)
        
        ## Check the population of intensity. Empty bins can be in three places: start edge, end edge, and in the middle.
        # The two former is to be corrected and the last throw a warning
        zeros = summedMonitor == 0
        if np.sum(zeros) != 0:
            diff = np.abs(np.diff(zeros))!=0
            diffIdx = np.arange(len(diff))[diff]
            

            emptyEndBins = zeros[diffIdx[-1]+1:]
            emptyStartBins = zeros[:diffIdx[0]+1]
            
            if np.all(emptyEndBins):
                emptyEndBins = -np.sum(emptyEndBins)-1
            else:
                emptyEndBins = -1
            
            if np.all(emptyStartBins):
                emptyStartBins = np.sum(emptyStartBins)
            else:
                emptyStartBins = 0

            if not np.sum(zeros) == emptyStartBins-emptyEndBins-1:
                warnings.warn('There are empty bins in the middle of the data! Please overlap these with additional detector settings or prepare to deal with 0 values.')
        else:
            emptyStartBins = 0
            emptyEndBins = -1

        normalizedIntensity = np.divide(summedRawIntensity[emptyStartBins:emptyEndBins],summedMonitor[emptyStartBins:emptyEndBins])
        normalizedIntensityError =  np.sqrt(summedRawIntensity[emptyStartBins:emptyEndBins])/summedMonitor[emptyStartBins:emptyEndBins]

        return twoThetaBins[emptyStartBins:emptyEndBins], normalizedIntensity, normalizedIntensityError,summedMonitor[emptyStartBins:emptyEndBins]
    

    @_tools.KwargChecker(function=plt.errorbar,include=_tools.MPLKwargs)
    def plotTwoTheta(self,ax=None,twoThetaBins=None,applyCalibration=True,correctedTwoTheta=True,dTheta=0.125,**kwargs):
        """Plot intensity as function of correctedTwoTheta or twoTheta
        Kwargs:
            - ax (axis): Matplotlib axis into which data is plotted (default None - generates new)
            - twoThetaBins (list): Bins into which 2theta is to be binned (default min(2theta),max(2theta) in steps of 0.1)
            - applyCalibration (bool): If true, take detector efficiency into account (default True)
            - correctedTwoTheta (bool): If true, use corrected two theta, otherwise sum vertically on detector (default True)
            - All other key word arguments are passed on to plotting routine
        Returns:
            - ax: Matplotlib axis into which data was plotted
            - twoThetaBins
            - normalizedIntensity
            - normalizedIntensityError
            - summedMonitor
        """
        
        
        twoThetaBins, normalizedIntensity, normalizedIntensityError,summedMonitor = self.sumDetector(twoThetaBins=twoThetaBins,applyCalibration=applyCalibration,\
                                                                                       correctedTwoTheta=correctedTwoTheta,dTheta=dTheta)
        TwoThetaPositions = 0.5*(twoThetaBins[:-1]+twoThetaBins[1:])

        if not 'fmt' in kwargs:
            kwargs['fmt'] = '-'

        if ax is None:
            fig,ax = plt.subplots()

        ax._errorbar = ax.errorbar(TwoThetaPositions,normalizedIntensity,yerr=normalizedIntensityError,**kwargs)
        ax.set_xlabel(r'$2\theta$ [deg]')
        ax.set_ylabel(r'Intensity [arb]')

        def format_coord(ax,xdata,ydata):
            if not hasattr(ax,'xfmt'):
                ax.mean_x_power = _tools.roundPower(np.mean(np.diff(ax._errorbar.get_children()[0].get_data()[0])))
                ax.xfmt = r'$2\theta$ = {:3.'+str(ax.mean_x_power)+'f} Deg'
            if not hasattr(ax,'yfmt'):
                ymin,ymax,ystep = [f(ax._errorbar.get_children()[0].get_data()[1]) for f in [np.min,np.max,len]]
                
                ax.mean_y_power = _tools.roundPower((ymax-ymin)/ystep)
                ax.yfmt = r'Int = {:.'+str(ax.mean_y_power)+'f} cts'

            return ', '.join([ax.xfmt.format(xdata),ax.yfmt.format(ydata)])

        ax.format_coord = lambda format_xdata,format_ydata:format_coord(ax,format_xdata,format_ydata)

        return ax,twoThetaBins, normalizedIntensity, normalizedIntensityError,summedMonitor

    def Viewer3D(self,dqx,dqy,dqz,rlu=True,axis=2, raw=False,  log=False, grid = True, outputFunction=print, 
                 cmap='viridis', steps=None, multiplicationFactor=1):

        """Generate a 3D view of all data files in the DatSet.
        
        Args:
            - dqx (float): Bin size along first axis in 1/AA
            - dqy (float): Bin size along second axis in 1/AA
            - dqz (float): Bin size along third axis in 1/AA
        Kwargs:
            - rlu (bool): Plot using reciprocal lattice units (default False)
            - axis (int): Initial view direction for the viewer (default 2)
            - raw (bool): If True plot counts else plot normalized counts (default False)
            - log (bool): Plot intensity as logarithm of intensity (default False)
            - grid (bool): Plot a grid on the figure (default True)
            - outputFunction (function): Function called when clicking on the figure (default print)
            - cmap (str): Name of color map used for plot (default viridis)
            - multiplicationFactor (float): Multiply intensities with this factor (default 1)
        """
        if rlu:
            
            QxQySample = copy.deepcopy(self.sample[0])
            QxQzSample = copy.deepcopy(self.sample[0])
            QyQzSample = copy.deepcopy(self.sample[0])
            
            samples = [QxQySample,QxQzSample,QyQzSample]
            projections = [[0,1,2],
                           [2,0,1],
                           [1,2,0]]
            
            axes = []
            figure = None
            for sample,proj in zip(samples,projections):
                p1,p2,p3 = _tools.findOrthogonalBasis(*sample.projectionVectors.T, sample.B)[proj]
                points = [np.dot(self.sample[0].UB,p) for p in [[0.0,0.0,0.0],p1,p2]]

                rot,trans = _tools.calculateRotationMatrixAndOffset2(points)
                
                sample.P1 = p1
                sample.P2 = p2
                sample.P3 = p3
                sample.projectionVectors = np.array([sample.P1,sample.P2,sample.P3]).T
                sample.ROT = rot

                ax = self.createRLUAxes(figure=figure,sample=sample)#**kwargs)
                axes.append(ax)
                
                figure = ax.get_figure()
                figure.delaxes(ax)

            axes = np.asarray(axes,dtype=object)[[2,1,0]]#[rluAxesQyQz,rluAxesQxQz,rluAxesQxQy]

        else:
            axes = None

        Data,bins,_ = self.binData3D(dqx,dqy,dqz,rlu=rlu,raw=raw,steps=steps)

        Data*=multiplicationFactor

        return Viewer3D.Viewer3D(Data,bins,axis=axis, ax=axes, grid=grid, log=log, outputFunction=outputFunction, cmap=cmap)
    
    def binData3D(self,dqx,dqy,dqz,rlu=True,raw=False,steps=10):
        """
        Bin scattering data in equi-sized bins.

        Args:
            - dqx (float): bin size along x in 1/AA
            - dqy (float): bin size along y in 1/AA
            - dqz (float): bin size along z in 1/AA
        Kwargs:
            - rlu (bool): flag to choose if data is rotated into rlu or kept in the instrument coordinate system (default True)
            - raw (bool): if True, keep scattering numbers un-normalized (default false)
            - steps (int): number of simultaneously treated scan steps (default 10)
        Returns:
            - Intensities (float): Scattering intensity
            - bins (float): bin edges in 3D
            - Errors (float): Errors corresponding ot the intensities
        """
        maximas = []
        minimas = []
        for df in self:
            if rlu:
                pos = np.einsum('ij,jk',df.sample.ROT,df.q[None].reshape(3,-1))
            else:
                pos = df.q[None].reshape(3,-1)
            maximas.append(np.max(pos,axis=1))
            minimas.append(np.min(pos,axis=1))

        maximas = np.max(maximas,axis=0)
        minimas = np.min(minimas,axis=0)
        extremePositions = np.array([minimas,maximas]).T
        bins = _tools.calculateBins(dqx,dqy,dqz,extremePositions)

        returndata = None
        for df in self:
            
            if steps is None:
                steps = len(df)
            
            stepsTaken = 0

                
            for idx in _tools.arange(0,len(df),steps):
                q = df.q[idx[0]:idx[1]]
                if raw:
                    dat = df.countsSliced(slice(idx[0],idx[1]))
                else:
                    dat = df.intensitySliced(slice(idx[0],idx[1]))
                    

                mon = df.monitor[idx[0]:idx[1]]
                mon=np.repeat(np.repeat(mon[:,np.newaxis],dat.shape[1],axis=1)[:,:,np.newaxis],dat.shape[2],axis=-1)
                
                print(df.fileName,'from',idx[0],'to',idx[-1])
                stepsTaken+=steps

                if rlu:
                    pos = np.einsum('ij,j...',df.sample.ROT,q).transpose(0,3,1,2) # shape -> steps,3,128,1152

                else:
                    pos = q.transpose(1,0,2,3)# shape -> steps,3,128,1152

                if True:
                    pos = pos.transpose(1,0,2,3)
                    boolMask = np.logical_not(df.mask[idx[0]:idx[1]].flatten())
                    localReturndata,_ = _tools.binData3D(dqx,dqy,dqz,pos=pos.reshape(3,-1)[:,boolMask],data=dat.flatten()[boolMask],mon=mon.flatten()[boolMask],bins = bins)

                    if returndata is None:
                        returndata = localReturndata
                    else:

                        
                        for data,newData in zip(returndata,localReturndata):
                            data+=newData
                    
        with warnings.catch_warnings() as w:
            warnings.simplefilter("ignore")
            intensities = np.divide(returndata[0],returndata[1])
            errors = np.divide(np.sqrt(returndata[0]),returndata[1])
        NaNs = returndata[-1]==0
        intensities[NaNs]=np.nan
        errors[NaNs]=np.nan
        return intensities,bins,errors

    @_tools.KwargChecker()
    @_tools.overWritingFunctionDecorator(RLUAxes.createRLUAxes)
    def createRLUAxes(*args,**kwargs): # pragma: no cover
        raise RuntimeError('This code is not meant to be run but rather is to be overwritten by decorator. Something is wrong!! Should run {}'.format(RLUAxes.createRLUAxes))

        
    def plotCut1D(self,P1,P2,rlu=True,stepSize=0.01,width=0.05,widthZ=0.05,raw=False,optimize=True,ax=None,steps=None,**kwargs):
        """Cut and plot data from P1 to P2 in steps of stepSize [1/AA] width a cylindrical width [1/AA]
        Args:
            - P1 (list): Start position for cut in either (Qx,Qy,Qz) or (H,K,L)
            - P2 (list): End position for cut in either (Qx,Qy,Qz) or (H,K,L)
        Kwargs:
            - rlu (bool): If True, P1 and P2 are in HKL, otherwise in QxQyQz (default True)
            - stepSize (float): Size of bins along cut direction in units of [1/AA] (default 0.01)
            - width (float): Integration width orthogonal to cut in units of [1/AA] (default 0.02)
            - raw (bool): If True, do not normalize data (default False)
            - optimize (bool): If True, perform optimized cutting (default True)
            - ax (matplotlib.axes): If None, a new is created (default None)
            - kwargs: All other kwargs are provided to the errorbar plot of the axis
        Returns:
            - Pos,Int,Ax
        """

        hkl,I,err = self.cut1D(P1=P1,P2=P2,rlu=rlu,stepSize=stepSize,width=width,widthZ=widthZ,raw=raw,optimize=optimize,steps=steps)
        if ax is None:
            ax  = generate1DAxis(P1,P2,rlu=rlu)

        if hasattr(ax,'calculatePositionInv'):
            X = ax.calculatePositionInv(*hkl)
        else:
            X = np.linalg.norm(*hkl,axis=1)

        if not 'fmt' in kwargs:
            kwargs['fmt'] = 'o'
            
        ax.errorbar(X,I,yerr=err,**kwargs)

        ax.get_figure().tight_layout()
        return hkl,I,err,ax

    def cut1D(self,P1,P2,rlu=True,stepSize=0.01,width=0.05,widthZ=0.05,raw=False,optimize=True,steps=None):
        """Cut data from P1 to P2 in steps of stepSize [1/AA] width a cylindrical width [1/AA]
        Args:
            - P1 (list): Start position for cut in either (Qx,Qy,Qz) or (H,K,L)
            - P2 (list): End position for cut in either (Qx,Qy,Qz) or (H,K,L)
        Kwargs:
            - rlu (bool): If True, P1 and P2 are in HKL, otherwise in QxQyQz (default True)
            - stepSize (float): Size of bins along cut direction in units of [1/AA] (default 0.01)
            - width (float): Integration width orthogonal to cut in units of [1/AA] (default 0.02)
            - raw (bool): If True, do not normalize data (default False)
            - optimize (bool): If True, perform optimized cutting (default True)
        Returns:
            - Pos,Int, Errors
            
        """
        intensities = None
        for df in self:
            if rlu:
                QStart = df.sample.calculateHKLToQxQyQz(*P1)
                QStop  = df.sample.calculateHKLToQxQyQz(*P2)
            else:
                QStart = P1
                QStop  = P2
            
            
            directionVector = (QStop-QStart).reshape(3,1)
            length = np.linalg.norm(directionVector)
            if np.isclose(length,0.0):
                raise AttributeError('The vector connecting the cut points has length 0. Received P1={}, P2={}'.format(','.join([str(x) for x in P1]),','.join([str(x) for x in P2])))
            directionVector*=1.0/length
            
            stopAlong = np.dot(QStop-QStart,directionVector)[0]
            sign = np.sign(stopAlong)
            
            bins = np.arange(-stepSize*0.5,np.abs(stopAlong)+stepSize*0.51,stepSize)
            
            intensity = []
            pos = []
            if steps is None:
                steps = len(df)
            for idx in _tools.arange(0,len(df),steps):
                print(df.fileName,'from',idx[0],'to',idx[-1])
                if not raw:
                    data = df.intensitySliced(slice(idx[0],idx[1]))
                else:
                    data = df.countsSliced(slice(idx[0],idx[1]))
                if optimize:
                
                    optimizationStepInPlane = 0.05
                    optimizationStepInPlane = np.min([optimizationStepInPlane,width*0.6])

                    ## Define boundig box
                    direction = QStop-QStart
                    directionLength=np.linalg.norm(direction)
                    direction*=1.0/directionLength
                    orthogonal = np.cross(direction,np.array([0,0,1]))
                    orthogonalVertical = np.cross(direction,orthogonal)


                    # Factor between actual cut and width used for cutoff
                    expansionFactor = 1.15
                    effectiveWidth = expansionFactor*width
                        
                    startEdge = QStart.reshape(3,1)+np.arange(-effectiveWidth*0.5,effectiveWidth*0.51,optimizationStepInPlane).reshape(1,-1)*orthogonal.reshape(3,1)-stepSize*direction.reshape(3,1)
                    endEdge = QStop.reshape(3,1)+np.arange(-effectiveWidth*0.5,effectiveWidth*0.51,optimizationStepInPlane).reshape(1,-1)*orthogonal.reshape(3,1)+stepSize*direction.reshape(3,1)
                    rightEdge = QStart.reshape(3,1)+0.5*effectiveWidth*orthogonal.reshape(3,1)+np.arange(-stepSize,directionLength+stepSize,optimizationStepInPlane)*direction.reshape(3,1)
                    leftEdge =  QStart.reshape(3,1)-0.5*effectiveWidth*orthogonal.reshape(3,1)+np.arange(-stepSize,directionLength+stepSize,optimizationStepInPlane)*direction.reshape(3,1)

                    voff = (effectiveWidth*0.5*orthogonalVertical).reshape(3,1)

                    checkPositions = np.concatenate([startEdge+voff,endEdge+voff,
                                                    rightEdge+voff,leftEdge+voff,
                                                    startEdge-voff,endEdge-voff,
                                                    rightEdge-voff,leftEdge-voff],axis=1)
                    # Calculate the corresponding A3, A4, and Z positions

                    A3,A4,Z = np.array([TasUBlibDEG.converterToA3A4Z(*pos,df.Ki,df.Ki,A4Sign=-1,radius=df.radius) for pos in checkPositions.T]).T

                    # remove nan-values
                    A4NonNaN = np.logical_not(np.isnan(A4))
                    A3 = A3[A4NonNaN]
                    A4 = A4[A4NonNaN]
                    Z = Z[A4NonNaN]
                    

                    A3Min,A3Max = [f(A3) for f in [np.nanmin,np.nanmax]]
                    A4Min,A4Max = [f(A4) for f in [np.nanmin,np.nanmax]]
                    ZMin,ZMax = [f(Z) for f in [np.nanmin,np.nanmax]]

                    # Find and sort ascending the indices        
                    twoThetaInside = np.logical_and(df.twoTheta[0]>A4Min,df.twoTheta[0]<A4Max)
                    A3Inside = np.logical_and(df.A3[idx[0]:idx[1]]>A3Min,df.A3[idx[0]:idx[1]]<A3Max)
                    ZInside = np.logical_and(df.verticalPosition>ZMin,df.verticalPosition<ZMax)


                    mask = np.zeros_like(data,dtype=bool)
                    mask[A3Inside,:,:]=True
                    mask[:,:,twoThetaInside]=True
                    mask[:,ZInside,:]=True

                    data = data[mask]
                    relativePosition = df.q[idx[0]:idx[1]][:,mask]-QStart.reshape(3,-1)

                    
                else:
                    relativePosition = df.q[idx[0]:idx[1]].reshape(3,-1)-QStart.reshape(3,-1)
                
                along = np.einsum('ij,i...->...j',relativePosition,directionVector)
                    
                orthogonal = np.linalg.norm(relativePosition-along*directionVector,axis=0)
                
                orthogonal = np.linalg.norm(relativePosition-along*directionVector,axis=0)
                test1 = (orthogonal<width*0.5).flatten()
                test2 = (along[0]>-stepSize).flatten()
                test3 = (along[0]<np.linalg.norm(stopAlong)+stepSize).flatten()
                
                insideQ = np.all([test1,test2,test3],axis=0)
            
                intensity = data.flatten()[insideQ]
                pos = sign*along.flatten()[insideQ]

                    
                weights = [intensity]
                _intensities,_normCounts = _tools.histogramdd(pos.reshape(-1,1),bins=[bins],weights=weights,returnCounts=True)
                _monitors = np.full_like(_intensities,df.monitor[0])
                if intensities is None:
                    intensities,normCounts,monitors = _intensities,_normCounts,_monitors
                else:
                    intensities+=_intensities
                    normCounts+=_normCounts
                    monitors += _monitors
            
        I = np.divide(intensities,monitors)
        errors = np.divide(np.sqrt(intensities),monitors)
        I[normCounts==0]=np.nan
        I[errors==0]=np.nan
        binCentres = 0.5*(bins[:-1]+bins[1:])
        
        positionVector = directionVector*binCentres+QStart.reshape(3,1)
        if rlu:
            positionVector = np.array([self[-1].sample.calculateQxQyQzToHKL(*bC) for bC in positionVector.T]).T
        return positionVector,I,errors


    def saveSampleToDisk(self,fileName=None,dataFolder=None):

        """
        function to store sample object from a df into a binary file.
        Kwargs:
            - fileName (str): fileName for UB matrix file. Default is None and sample name form the ds
            - dataFolder (str): directory for saving UB file. Default is None, and in current working directory
        """

        if fileName is None:
            fileName = self.sample[0].name 

        if dataFolder is None:
            dataFolder = os.getcwd()  
        
        filePath = os.path.join(dataFolder,fileName)
        
        _tools.saveSampleToDesk(self[0].sample,str(filePath))


    def loadSample(self,filePath):
        """
        function to load UB from binary file into all dataFiles in a dataSet.
        args: 
            filePath (str): Filepath to UB matrix
            
        """
        sampleLoaded = _tools.loadSampleFromDesk(filePath)
        
        for df in self:

            df.sample.ROT = sampleLoaded.ROT
            df.sample.P1  = sampleLoaded.P1 
            df.sample.P2  = sampleLoaded.P2 
            df.sample.P3  = sampleLoaded.P3 

            df.sample.offsetA3 = sampleLoaded.offsetA3
            df.sample.RotationToScatteringPlane = sampleLoaded.RotationToScatteringPlane
            df.sample.foundPeakPositions = sampleLoaded.foundPeakPositions

            df.sample.projectionVectors = sampleLoaded.projectionVectors
            
            df.sample.projectionB = sampleLoaded.projectionB
            df.sample.UB = sampleLoaded.UB
            
            df.sample.peakUsedForAlignment = sampleLoaded.peakUsedForAlignment
        
        print('UB loaded')



    def autoAlignScatteringPlane(self,scatteringNormal,threshold=30,dx=0.04,dy=0.04,dz=0.08,distanceThreshold=0.15):
        """Automatically align scattering plane and peaks within
        
        Args:
            
            - scatteringNormal (vector 3D): Normal to the scattering plane in HKL
            
            
        Kwargs:
            
            - threshold (float): Thresholding for intensities within the 3D binned data used in peak search (default 30)
            
            - dx (float): size of 3D binning along Qx (default 0.04)
            
            - dy (float): size of 3D binning along Qy (default 0.04)
            
            - dz (float): size of 3D binning along Qz (default 0.08)
            
            - distanceThreshold (float): Distance in 1/AA where peaks are clustered together (default 0.15)
          
            
        This methods is an attempt to automatically align the scattering plane of all data files
        within the DataSet. The algorithm works as follows for each data file individually :
            
            1) Perform a 3D binning of data in to equi-sized bins with size (dx,dy,dz)
            
            2) "Peaks" are defined as all positions having intensities>threshold
        
            3) These peaks are clustered together if closer than 0.02 1/AA and centre of gravity
               using intensity is applied to find common centre.
               
            4) Above step is repeated with custom distanceThreshold
        
            5) Plane normals are found as cross products between all vectors connecting 
               all found peaks -> Gives a list of approximately NPeaks*(NPeaks-1)*(NPeaks-2)
               
            6) Plane normals are clustered and most common is used
        
            7) Peaks are rotated into the scattering plane
        
            8) Within the plane all found peaks are projected along the 'nice' plane vectors and
               the peak having the scattering length closest to an integer multiple of either is
               chosen for alignment
               
            9) Rotation within the plane is found by rotating the found peak either along x or y 
               depending on which projection vector was closest
               
           10) Sample is updated with found rotations.
        
        
        On the sample object, the peak used for alignment within the scattering plane is
        saved as sample.peakUsedForAlignment as a dictionary with 'HKL' and 'QxQyQz' holding
        suggested HKL point and original QxQyQz position.
        """
        peakPositions = []
        peakWeights = []
        for df in self:
            
            # 1) 
            Intensities,bins = _tools.binData3D(dx,dy,dz,df.q[None].reshape(3,-1),df.intensity)
            with warnings.catch_warnings() as w:
                warnings.simplefilter("ignore")
                Intensities = np.divide(Intensities[0],Intensities[1])
            
            
            # 2)
            possiblePeaks = Intensities>threshold
            
            ints = Intensities[possiblePeaks]
            
            centerPoints = [b[:-1,:-1,:-1]+0.5*dB for b,dB in zip(bins,[dx,dy,dz])]
            
            positions = np.array([b[possiblePeaks] for b in centerPoints]).T
            
            # 3) assuming worse resolution out of plane
            distanceFunctionLocal = lambda a,b: _tools.distance(a,b,dx=1.0,dy=1.0,dz=0.5)
            peaksInitial = _tools.clusterPoints(positions,ints,distanceThreshold=0.02,distanceFunction=distanceFunctionLocal) 
            
            if len(peaksInitial) == 0:
                continue
            peakPositions.append(list(p.position for p in peaksInitial))
            peakWeights.append(list(p.weight for p in peaksInitial))
            print('{:} peaks found in '.format(len(ints)),df.fileName)
            

        peakPositions = np.concatenate(peakPositions)
        peakWeights = np.concatenate(peakWeights)
        # 4) 
        self.peaks = _tools.clusterPoints(peakPositions,peakWeights,distanceThreshold=distanceThreshold,distanceFunction=distanceFunctionLocal)
        
        
        foundPeakPositions = np.array([p.position for p in self.peaks])
        
        # 5) Make list of triplet normals, e.g. cross products between all vectors connecting all found peaks
        tripletNormal = _tools.calculateTriplets(foundPeakPositions,normalized=True)
        
        
        # 6) Combine the normal vectors closest to each other.
        normalVectors = _tools.clusterPoints(tripletNormal,np.ones(len(tripletNormal)),distanceThreshold=0.01)
        
        # Find the most frequent normal vector as the one with highest weight
        bestNormalVector = normalVectors[np.argmax([p.weight for p in normalVectors])].position
        
        # 7) 
        # Find rotation matrix transforming bestNormalVector to lay along the z-axis
        # Rotation is performed around the vector perpendicular to bestNormalVector and z-axis
        rotationVector = np.cross([0,0,1.0],bestNormalVector)
        rotationVector*=1.0/np.linalg.norm(rotationVector)
        # Rotation angle is given by the regular cosine relation, but due to z being [0,0,1] and both normal
        
        theta = np.arccos(bestNormalVector[2]) # dot(bestNormalVector,[0,0,1])/(lengths) <-- both unit vectors
        
        RotationToScatteringPlane = _tools.rotMatrix(rotationVector, theta,deg=False)
            
        # Rotated all found peaks to the scattering plane
        rotatedPeaks = np.einsum('ji,...j->...i',RotationToScatteringPlane,foundPeakPositions)
        
        # 8) Start by finding in plane vectors using provided scattering plane normal
        scatteringNormal = _tools.LengthOrder(np.asarray(scatteringNormal,dtype=float)) # 
        
        # Calculate into Q
        scatteringNormalB = np.dot(df.sample.B,scatteringNormal)
        
        # Find two main "nice" vectors in the scattering plane by finding a vector orthogonal to scatteringNormalB
        InPlaneGuess = np.cross(scatteringNormalB,np.array([1,-1,0]))
        
        # If scatteringNormalB happens to be along [1,-1,0] we just try again!
        if np.isclose(np.linalg.norm(InPlaneGuess),0.0,atol=1e-3):
            InPlaneGuess = np.cross(scatteringNormalB,np.array([1,1,0]))
            
        # planeVector1 is ensured to be orthogonal to scatteringNormalB 
        # (rounding is needed to better beautify the vector)
        planeVector1 = np.round(np.cross(InPlaneGuess,scatteringNormalB),4)
        planeVector1 = _tools.LengthOrder(planeVector1)

        # Second vector is radially found orthogonal to scatteringNormalB and planeVector1
        planeVector2 = np.round(np.cross(scatteringNormalB,planeVector1),4)
        planeVector2 = _tools.LengthOrder(planeVector2)
        
        # Try to align the last free rotation within the scattering plane by calculating
        # the length of the scattering vectors for the peaks and compare to moduli of 
        # planeVector1 and planeVector2
        
        lengthPeaksInPlane = np.linalg.norm(rotatedPeaks,axis=1)
        
        planeVector1Length = np.linalg.norm(np.dot(df.sample.B,planeVector1))
        planeVector2Length = np.linalg.norm(np.dot(df.sample.B,planeVector2))
        
        projectionAlongPV1 = lengthPeaksInPlane/planeVector1Length
        projectionAlongPV2 = lengthPeaksInPlane/planeVector2Length
        
        # Initialize the along1 and along2 with False to force while loop
        along1 = [False]
        along2 = [False]
        
        atol = 0.001
        # While there are no peaks found along with a modulus close to 0/1 along the main directions 
        # iteratively increase tolerance (atol)
        while np.sum(along1)+np.sum(along2) == 0: 
            atol+=0.001
            along1 = np.array([np.logical_or(np.isclose(x,0.0,atol=atol),np.isclose(x,1.0,atol=atol)) for x in np.mod(projectionAlongPV1,1)])
            along2 = np.array([np.logical_or(np.isclose(x,0.0,atol=atol),np.isclose(x,1.0,atol=atol)) for x in np.mod(projectionAlongPV2,1)])
        
        # Either we found a peak along planeVector1 or planeVector2
        if np.sum(along1)> 0:
            foundPosition = rotatedPeaks[along1][0]
            axisOffset = 0.0 # We want planeVector1 to be along the x-axis
            peakUsedForAlignment = {'HKL':   planeVector1*projectionAlongPV1[along1][0],
                                    'QxQyQz':foundPeakPositions[along1][0]}
        
        else:
            foundPosition = rotatedPeaks[along2][0]
            axisOffset = 90.0 # planeVector2 is along the y-axis, e.i. 90 deg rotated from x
            peakUsedForAlignment = {'HKL':   planeVector2*projectionAlongPV2[along2][0],
                                    'QxQyQz':foundPeakPositions[along2][0]}
        
        # 9) 
        # Calculate the actual position of the peak found along planeVector1 or planeVector2
        offsetA3 = np.rad2deg(np.arctan2(foundPosition[1],foundPosition[0]))-axisOffset

        # Find rotation matrix which is around the z-axis and has angle of -offsetA3
        rotation = np.dot(_tools.rotMatrix(np.array([0,0,1.0]),-offsetA3),RotationToScatteringPlane.T)
        
        # 10) 
        # sample rotation has now been found (converts between instrument 
        # qx,qy,qz to qx along planeVector1 and qy along planeVector2)
        for df in self:
            sample = df.sample
            sample.ROT = rotation
            sample.P1 = _tools.LengthOrder(planeVector1)
            sample.P2 = _tools.LengthOrder(planeVector2)
            sample.P3 = _tools.LengthOrder(scatteringNormal)

            sample.offsetA3 = offsetA3
            sample.RotationToScatteringPlane = RotationToScatteringPlane
            sample.foundPeakPositions = foundPeakPositions

            sample.projectionVectors = np.array([sample.P1,sample.P2,sample.P3]).T
            
            sample.projectionB = np.diag(np.linalg.norm(np.dot(sample.projectionVectors.T,sample.B),axis=1))
            sample.UB = np.dot(sample.ROT.T,np.dot(sample.projectionB,np.linalg.inv(sample.projectionVectors)))
            
            sample.peakUsedForAlignment = peakUsedForAlignment


    
    def autoAlignToRef(self,scatteringNormal,inPlaneRef=None,planeVector2=None,threshold=30,dx=0.04,dy=0.04,dz=0.08,distanceThreshold=0.15,axisOffset=0.0):
        """Automatically align scattering plane and peaks within
        
        Args:
            
            - scatteringNormal (vector 3D): Normal to the scattering plane in HKL
            
            
        Kwargs:
            
            - threshold (float): Thresholding for intensities within the 3D binned data used in peak search (default 30)
            
            - dx (float): size of 3D binning along Qx (default 0.04)
            
            - dy (float): size of 3D binning along Qy (default 0.04)
            
            - dz (float): size of 3D binning along Qz (default 0.08)
            
            - distanceThreshold (float): Distance in 1/AA where peaks are clustered together (default 0.15)
          
            
        This methods is an attempt to automatically align the scattering plane of all data files
        within the DataSet. The algorithm works as follows for each data file individually :
            
            1) Perform a 3D binning of data in to equi-sized bins with size (dx,dy,dz)
            
            2) "Peaks" are defined as all positions having intensities>threshold
        
            3) These peaks are clustered together if closer than 0.02 1/AA and centre of gravity
               using intensity is applied to find common centre.
               
            4) Above step is repeated with custom distanceThreshold
        
            5) Plane normals are found as cross products between all vectors connecting 
               all found peaks -> Gives a list of approximately NPeaks*(NPeaks-1)*(NPeaks-2)
               
            6) Plane normals are clustered and most common is used
        
            7) Peaks are rotated into the scattering plane
        
            8) Within the plane all found peaks are projected along the 'nice' plane vectors and
               the peak having the scattering length closest to an integer multiple of either is
               chosen for alignment
               
            9) Rotation within the plane is found by rotating the found peak either along x or y 
               depending on which projection vector was closest
               
           10) Sample is updated with found rotations.
        
        
        On the sample object, the peak used for alignment within the scattering plane is
        saved as sample.peakUsedForAlignment as a dictionary with 'HKL' and 'QxQyQz' holding
        suggested HKL point and original QxQyQz position.
        
        """
        peakPositions = []
        peakWeights = []
        for df in self:
            
            # 1) 
            Intensities,bins = _tools.binData3D(dx,dy,dz,df.q[None].reshape(3,-1),df.intensity)
            with warnings.catch_warnings() as w:
                warnings.simplefilter("ignore")
                Intensities = np.divide(Intensities[0],Intensities[1])
            
            # 2)
            possiblePeaks = Intensities>threshold
            ints = Intensities[possiblePeaks]
            
            centerPoints = [b[:-1,:-1,:-1]+0.5*dB for b,dB in zip(bins,[dx,dy,dz])]
            
            positions = np.array([b[possiblePeaks] for b in centerPoints]).T
            
            # 3) assuming worse resolution out of plane
            distanceFunctionLocal = lambda a,b: _tools.distance(a,b,dx=1.0,dy=1.0,dz=0.5)
            peaksInitial = _tools.clusterPoints(positions,ints,distanceThreshold=0.02,distanceFunction=distanceFunctionLocal) 
            
            peakPositions.append(list(p.position for p in peaksInitial))
            peakWeights.append(list(p.weight for p in peaksInitial))
            print('{:} peaks found in '.format(len(ints)),df.fileName)

        peakPositions = np.concatenate(peakPositions)
        peakWeights = np.concatenate(peakWeights)
        # 4) 
        peaks = _tools.clusterPoints(peakPositions,peakWeights,distanceThreshold=distanceThreshold,distanceFunction=distanceFunctionLocal)
        
        foundPeakPositions = np.array([p.position for p in peaks])
        
        # 5) Make list of triplet normals, e.g. cross products between all vectors connecting all found peaks
        tripletNormal = _tools.calculateTriplets(foundPeakPositions,normalized=True)
        
        
        # 6) Combine the normal vectors closest to each other.
        normalVectors = _tools.clusterPoints(tripletNormal,np.ones(len(tripletNormal)),distanceThreshold=0.01)
        
        # Find the most frequent normal vector as the one with highest weight
        bestNormalVector = normalVectors[np.argmax([p.weight for p in normalVectors])].position
        
        # 7) 
        # Find rotation matrix transforming bestNormalVector to lay along the z-axis
        # Rotation is performed around the vector perpendicular to bestNormalVector and z-axis
        rotationVector = np.cross([0,0,1.0],bestNormalVector)
        rotationVector*=1.0/np.linalg.norm(rotationVector)
        # Rotation angle is given by the regular cosine relation, but due to z being [0,0,1] and both normal
        
        theta = np.arccos(bestNormalVector[2]) # dot(bestNormalVector,[0,0,1])/(lengths) <-- both unit vectors
        
        RotationToScatteringPlane = _tools.rotMatrix(rotationVector, theta,deg=False)
            
        # Rotated all found peaks to the scattering plane
        rotatedPeaks = np.einsum('ji,...j->...i',RotationToScatteringPlane,foundPeakPositions)
        
        # 8) Start by finding in plane vectors using provided scattering plane normal
        scatteringNormal = _tools.LengthOrder(np.asarray(scatteringNormal,dtype=float)) # 
        
        # Calculate into Q
        scatteringNormalB = np.dot(df.sample.B,scatteringNormal)
        
        # Find two main "nice" vectors in the scattering plane from input
        if inPlaneRef is None:
            print('no inPLaneRef given')
        else:
            planeVector1 = inPlaneRef # or P1???
        
        # Second vector is given as input
        if planeVector2 is None:
            print('no planeVector2 given')
            planeVector2 = np.round(np.cross(scatteringNormalB,planeVector1),4)
            planeVector2 = _tools.LengthOrder(planeVector2)
            print('planeVector2 found to be: ', planeVector2)
        
        # Try to align the last free rotation within the scattering plane by calculating
        # the length of the scattering vectors for the peaks and compare to moduli of 
        # planeVector1 
        
        lengthPeaksInPlane = np.linalg.norm(rotatedPeaks,axis=1)
        
        planeVector1Length = np.linalg.norm(np.dot(df.sample.B,planeVector1))
        
        projectionAlongPV1 = lengthPeaksInPlane/planeVector1Length
        
        # Initialize the along1 and along2 with False to force while loop
        along1 = [False]
        
        atol = 0.001
        # While there are no peaks found along with a modulus close to 0/1 along the main directions 
        # iteratively increase tolerance (atol)
        while np.sum(along1) == 0: 
            atol+=0.001
            along1 = np.array([np.logical_or(np.isclose(x,0.0,atol=atol),np.isclose(x,1.0,atol=atol)) for x in np.mod(projectionAlongPV1,1)])
    
        foundPosition = rotatedPeaks[along1][0]
        
        peakUsedForAlignment = {'HKL':   planeVector1*projectionAlongPV1[along1][0],
                                'QxQyQz':foundPeakPositions[along1][0]}
        
        # 9) 
        # Calculate the actual position of the peak found along planeVector1 
        offsetA3 = np.rad2deg(np.arctan2(foundPosition[1],foundPosition[0]))-axisOffset

        # Find rotation matrix which is around the z-axis and has angle of -offsetA3
        rotation = np.dot(_tools.rotMatrix(np.array([0,0,1.0]),-offsetA3),RotationToScatteringPlane.T)
        
        # 10) 
        # sample rotation has now been found (converts between instrument 
        # qx,qy,qz to qx along planeVector1 and qy along planeVector2)
        for df in self:
            sample = df.sample
            sample.ROT = rotation
            sample.P1 = _tools.LengthOrder(planeVector1)
            sample.P2 = _tools.LengthOrder(planeVector2)
            sample.P3 = _tools.LengthOrder(scatteringNormal)

            sample.offsetA3 = offsetA3
            sample.RotationToScatteringPlane = RotationToScatteringPlane
            sample.foundPeakPositions = foundPeakPositions

            sample.projectionVectors = np.array([sample.P1,sample.P2,sample.P3]).T
            
            sample.projectionB = np.diag(np.linalg.norm(np.dot(sample.projectionVectors.T,sample.B),axis=1))
            sample.UB = np.dot(sample.ROT.T,np.dot(sample.projectionB,np.linalg.inv(sample.projectionVectors)))
            
            sample.peakUsedForAlignment = peakUsedForAlignment 


    def alignToRef(self,coordinates,planeVector1,planeVector2,optimize=False,axisOffset=0.0):
        """
        Args:
            
            - coordinates (list): peak position to align for planeVector1 in Qx, Qy, Qz
            - planeVector1 (list): Indicies of the reflection used for alignment (Or directional vector???)
            - planeVector2 (list): Vector along y axis
            
            
        Kwargs:
            
            - optimize = False (bool): Fit position of peak, default is False. NOT WORKING!
        
        This method takes coordinates of a reflection in Qz,Qy,Qz and align that peak to planeVector1. 
            1. find scattering normal from the plane vectors
            2. Find vector to rotate around from coordinates and scatteringNormal
            3. Find angle to rotate and rotation matrix
            4. Rotated peak to the scattering plane
            5. Find indices of reflection used for alignment
            6. calculate the actual position of the peak found along planeVector1 
            7. Find rotation matrix which is around the z-axis and has angle of -offsetA3
            8. sample rotation has now been found (converts between instrument  qx,qy,qz to qx along planeVector1 and qy along planeVector2)
            9. update sample
        """

        if optimize is True:
            # fit peak position
            pass
        else:
            pass


        df = self[0]

        # 1. find scattering normal from the plane vectors
        scatteringNormal = np.cross(planeVector1,planeVector2) 
        scatteringNormal = _tools.LengthOrder(scatteringNormal)

        # 2. Find vector to rotate around from coordinates and scatteringNormal
        rotationVector = np.cross(scatteringNormal,coordinates)
        rotationVector*=1.0/np.linalg.norm(rotationVector)

        # 3. Find angle to rotate and rotation matrix
        theta = np.round(np.arccos(np.linalg.norm(np.dot(scatteringNormal,coordinates))) - np.pi/2,5)
     
        RotationToScatteringPlane = _tools.rotMatrix(rotationVector, np.radians(theta), deg=False)
        
        # 4. Rotated peak to the scattering plane
        foundPosition = np.einsum('ji,...j->...i',RotationToScatteringPlane,coordinates)
        
        # 5. Find indices of reflection used for alignment
        lengthPeaksInPlane = np.linalg.norm(foundPosition)

        planeVector1Length = np.linalg.norm(np.dot(df.sample.B,planeVector1))
        
        projectionAlongPV1 = lengthPeaksInPlane/planeVector1Length

        peakUsedForAlignment = {'HKL':   planeVector1*projectionAlongPV1,
                                'QxQyQz': foundPosition}
        
        # 6. calculate the actual position of the peak found along planeVector1 
        offsetA3 = np.rad2deg(np.arctan2(foundPosition[1],foundPosition[0]))-axisOffset

        # 7. Find rotation matrix which is around the z-axis and has angle of -offsetA3
        rotation = np.dot(_tools.rotMatrix(np.array([0,0,1.0]),-offsetA3),RotationToScatteringPlane.T)
        
        # 8. sample rotation has now been found (converts between instrument  qx,qy,qz to qx along planeVector1 and qy along planeVector2)

        # 9. update sample
        for df in self:
            sample = df.sample
            sample.ROT = rotation
            sample.P1 = _tools.LengthOrder(planeVector1)
            sample.P2 = _tools.LengthOrder(planeVector2)
            sample.P3 = _tools.LengthOrder(scatteringNormal)

            sample.offsetA3 = offsetA3
            sample.RotationToScatteringPlane = RotationToScatteringPlane
            sample.foundPeakPositions = coordinates

            sample.projectionVectors = np.array([sample.P1,sample.P2,sample.P3]).T
            
            sample.projectionB = np.diag(np.linalg.norm(np.dot(sample.projectionVectors.T,sample.B),axis=1))
            sample.UB = np.dot(sample.ROT.T,np.dot(sample.projectionB,np.linalg.inv(sample.projectionVectors)))
            
            sample.peakUsedForAlignment = peakUsedForAlignment 

    def alignToRefs(self,q1,q2,HKL1,HKL2):
        """Generate UB matrix from two Q-points with corresponding HKL values
        
        Args:

            - q1 (array): Position of peak 1 in 1/AA

            - q2 (array): Position of peak 2 in 1/AA

            - HKL1 (array): Position of peak 1 in RLU

            - HKL2 (array): Position of peak 2 in RLU

        """
        
        E = np.power(self[0].Ki/TasUBlibDEG.factorsqrtEK,2.0)
        
        # Find rotation that brings q1 and q2 into the scattering plane (qz=0)
        planeNormal1 = np.array([0.0,0.0,1.0])
        planeNormal2 = np.cross(q1,q2)
        planeNormal2*= 1/(np.linalg.norm(planeNormal2))
        rotVector = np.cross(planeNormal2,planeNormal1)
        rotVector*=1.0/np.linalg.norm(rotVector)
        rotAngle = _tools.vectorAngle(planeNormal1, planeNormal2)
        if np.isclose(np.mod(rotAngle,np.pi),0):
            rotMatrix = np.eye(3)
        else:
            rotMatrix = _tools.rotMatrix(rotVector, rotAngle,deg=False)

        q1Rotated,q2Rotated = [np.dot(rotMatrix,q) for q in [q1,q2]]

        A31,A41,_ = TasUBlibDEG.converterToA3A4Z(*q1Rotated,Ki=self[0].Ki,Kf=self[0].Ki,A4Sign=-1,radius=self[0].radius)
        A32,A42,_ = TasUBlibDEG.converterToA3A4Z(*q2Rotated,Ki=self[0].Ki,Kf=self[0].Ki,A4Sign=-1,radius=self[0].radius)

        # H K L A3 A4 sgu sgl Ei Ef
        R1 = [*HKL1, A31, A41, 0.0, 0.0, E, E]
        R2 = [*HKL2, A32, A42, 0.0, 0.0, E, E]
        newUB = TasUBlibDEG.calcTasUBFromTwoReflections(self[0].sample.fullCell, R1, R2)
        
        # Counter-rotate UB with the rotation matrix from above
        newUB = np.dot(rotMatrix.T,newUB)

        projectionVector1,projectionVector2,projectionVector3 = np.eye(3)
            
        pV1q = np.dot(newUB,projectionVector1)
        pV2q = np.dot(newUB,projectionVector2)
        
        points = np.asarray([[0.0,0.0,0.0],pV1q,pV2q])
        rot,tr = _tools.calculateRotationMatrixAndOffset2(points)
        
        for s in self.sample:
            
            s.UB = newUB
            s.P1 = projectionVector1
            s.P2 = projectionVector2
            s.P3 = projectionVector3
            s.ROT = rot
        
            s.projectionVectors = np.array([s.P1,s.P2,s.P3]).T


    def peakSearch(self,threshold=30,dx=0.04,dy=0.04,dz=0.08,distanceThreshold=0.15):
        """ Search for peaks in data set
          
        Kwargs:
            
            - threshold (float): Thresholding for intensities within the 3D binned data used in peak search (default 30)
            
            - dx (float): size of 3D binning along Qx (default 0.04)
            
            - dy (float): size of 3D binning along Qy (default 0.04)
            
            - dz (float): size of 3D binning along Qz (default 0.08)
            
            - distanceThreshold (float): Distance in 1/AA where peaks are clustered together (default 0.15)
          
            
        This methods is an attempt to automatically align the scattering plane of all data files
        within the DataSet. The algorithm works as follows for each data file individually :
            
            1) Perform a 3D binning of data in to equi-sized bins with size (dx,dy,dz)
            
            2) "Peaks" are defined as all positions having intensities>threshold
        
            3) These peaks are clustered together if closer than 0.02 1/AA and centre of gravity
               using intensity is applied to find common centre.
               
            4) Above step is repeated with custom distanceThreshold
        
        """
        peakPositions = []
        peakWeights = []
        for df in self:
            
            # 1) 
            Intensities,bins = _tools.binData3D(dx,dy,dz,df.q[None].reshape(3,-1),df.intensity)
            with warnings.catch_warnings() as w:
                warnings.simplefilter("ignore")
                Intensities = np.divide(Intensities[0],Intensities[1])
            
            # 2)
            possiblePeaks = Intensities>threshold
            
            ints = Intensities[possiblePeaks]
            
            centerPoints = [b[:-1,:-1,:-1]+0.5*dB for b,dB in zip(bins,[dx,dy,dz])]
            
            positions = np.array([b[possiblePeaks] for b in centerPoints]).T
            
            # 3) assuming worse resolution out of plane
            distanceFunctionLocal = lambda a,b: _tools.distance(a,b,dx=1.0,dy=1.0,dz=0.5)
            peaksInitial = _tools.clusterPoints(positions,ints,distanceThreshold=0.02,distanceFunction=distanceFunctionLocal,fileName=df.fileName) 
            
            if len(peaksInitial) == 0:
                continue
            peakPositions.append(list(p.position for p in peaksInitial))
            peakWeights.append(list(p.weight for p in peaksInitial))
            print('{:} peaks found in '.format(len(ints)),df.fileName)
            

        peakPositions = np.concatenate(peakPositions)
        peakWeights = np.concatenate(peakWeights)
        # 4) 
        self.peaks = _tools.clusterPoints(peakPositions,peakWeights,distanceThreshold=distanceThreshold,distanceFunction=distanceFunctionLocal,fileName=df.fileName)
        
        foundPeakPositions = np.array([p.position for p in self.peaks])

        lenghtInQ = np.array([np.linalg.norm(vec) for vec in foundPeakPositions])

        posIn2Theta = np.array([ 2 * np.rad2deg(np.arcsin( q * self[0].wavelength / (4 * np.pi) ))  for q in lenghtInQ])
        
        foundPeakDic = {}

        for peak in np.arange(len(posIn2Theta)):
            foundPeakDic[str(peak)] = {'foundPeakPositions' : foundPeakPositions[peak] , 'lenghtInQ' : lenghtInQ[peak] , 'posIn2Theta' : posIn2Theta[peak] }

        return foundPeakPositions, lenghtInQ, posIn2Theta, foundPeakDic

    def rotateAroundScatteringNormal(self,rotation=0.0):
        """
        Function to rotate a dataSet around the surface normal. Projection vectors are updated not change direction, e.g. stay along x and y axis
        rotation (float): angle to rotate dataSet around scattering plane normal
        """

        offsetA3 = self[0].sample.offsetA3 
        offsetA3 = offsetA3 + rotation
  
        RotationToScatteringPlane = self[0].sample.RotationToScatteringPlane

        rotation = np.dot(_tools.rotMatrix(np.array([0,0,1.0]),-offsetA3),RotationToScatteringPlane.T)

        for df in self:
            sample = df.sample
            sample.ROT = rotation
            sample.offsetA3 = offsetA3
            sample.UB = np.dot(sample.ROT.T,np.dot(sample.projectionB,np.linalg.inv(sample.projectionVectors)))


    def boxIntegration(self,peakDic,roi=True,saveFig=False,title=None,integrationList=None,closeFigures=False,plane=None):
        """

        boxIntegration creates a region of interest on the detector (roi) and sum all intensity in the roi for a range of A3. 
        Peak positions on the detector is defined by the user. 
        No optiization methods avaliable.

        peakDic has the form:

        peakDic = { }
        peakDic['300'] = {
                    'h' : 3,
                    'k' : 0,
                    'l' : 0,
                    'df' : 0,
                    'A3_center' : 103, 
                    'A3_minus' : 20, 
                    'A3_pluss' : 20, 
                    'tth' : 113.8, 
                    'tth_minus' : 1, 
                    'tth_pluss' : 1, 
                    'startZ' : 50, 
                    'stopZ' : 100, 
                    'vmin' : 0,
                    'vmax' : 0.002,
                  }

        '300' : is the name of the peak
        'h' : Miller index
        'k' : Miller index
        'l' : Miller index
        'df' : index of the dataFile in the dataSet where the peak is
        'A3_center' : Center position of the peak in A3
        'A3_minus' : range in frames on the lower bound side of the peak 
        'A3_pluss' : range in frames on the higher bound side of the peak
        'tth' : 2tw of the peak
        'tth_minus' : lower bound limit in 2th
        'tth_pluss' : lower bound limit in 2th 
        'startZ' : lower bound for pixels on detector
        'stopZ' : higher bound for pixels on detector
        'vmin' : vmin for box integration plot
        'vmax' : vmax for box integration plot
        
        roi=True, Plot the rois on the detector for all A3
        saveFig=False, Save the figures, give str to add into the name used for saving
        title=None, Title for A3 figure
        integrationList=None, if you only want to integrate some peaks in your dictonary, give a list with str for the peak name
        closeFigures=False, for closing figures after a peak is integrated

        returns peakDic with:
                peakDic[peak]['summed_counts'] = np.sum(counts)
                peakDic[peak]['peak_cut'] = [xdata,ydata]
                peakDic[peak]['fit'] = [H,x0,sigma,FWHM,integrated]
        
        """

        if integrationList is None:
            integrationList = []
            for peak in peakDic: 
                integrationList.append(peak)

        for peak in peakDic:
        
            if peak in integrationList:
                        
                df = self[peakDic[peak]['df']]   
                
                # vertical range in pixcel
                startZ = peakDic[peak]['startZ']
                stopZ = peakDic[peak]['stopZ']
                
                # # # peak position
                tth = np.abs(peakDic[peak]['tth'])
                tth_minus = peakDic[peak]['tth_minus']
                tth_pluss = peakDic[peak]['tth_pluss']
                
                # # # twoTheta range
                startThetaVal = -(tth - tth_minus)
                stopThetaVal = -(tth +  tth_pluss)
                
                startTheta = np.argmin(np.abs(df.twoTheta[64]-startThetaVal))
                stopTheta = np.argmin(np.abs(df.twoTheta[64]-stopThetaVal))
                
                # # # A3 range
                A3_center = peakDic[peak]['A3_center']
                A3_minus = peakDic[peak]['A3_minus']
                A3_pluss = peakDic[peak]['A3_pluss']
                
                # Find index of A3
                absolute_differences = np.abs(df.A3 - A3_center)
                A3_center = np.argmin(absolute_differences)
                
                startA3 = A3_center - A3_minus
                stopA3 = A3_center + A3_pluss
                
                a3StepDegrees=(max(df.A3)-min(df.A3))/len(df.A3)
                A3Steps = abs(stopA3-startA3)
                A3StepSign = np.sign(stopA3-startA3)
                sttRange = (stopA3-startA3)*a3StepDegrees*2
                
                sttStepDegreesToIndex = np.diff(df.twoTheta[65])[0]
                sttSteps = A3StepSign*sttRange/sttStepDegreesToIndex
                sttOffset = np.linspace(-sttSteps*0.5,sttSteps*0.5,A3Steps).astype(int)
                
                countsAllTwoTheta = df.intensity[startA3:stopA3,startZ:stopZ,:].sum(axis=(1))
                monitors = df.monitor[startA3:stopA3]
                
                counts = []
                
                for i,offset in enumerate(sttOffset):
                    counts.append(countsAllTwoTheta[i,startTheta+offset:stopTheta+offset].sum())
                
                counts = np.asarray(counts) / monitors 
                
                xdata = df.A3[startA3:stopA3]
                ydata = counts
                
                H, A, x0, sigma = gauss_fit(xdata, ydata)
                FWHM = 2.35482 * sigma
               
                #Now calculate more points for the plot
                step = 0.01
                plotx = []
                ploty = []
                
                for value in np.arange(min(xdata),max(xdata)+step,step):
                    plotx.append(value)
                    ploty.append(gauss(value, H, A, x0, sigma))
                
                fig,ax = plt.subplots()
                ax.plot(xdata, ydata, 'bo--', linewidth=1, markersize=6,label='data')
                ax.plot(plotx, ploty, 'r', label='fit')
                plt.xlabel('A3 [deg.]')
                plt.ylabel('Intensity [arb. units]')

                if title is not None:
                    if plane is not None:
                        plt.title(f'{title} - {peak} in {plane}')
                    else:
                        plt.title(f'{title} - {peak}')
                
                if saveFig is not False:
                        fig.savefig(saveFig+f'{peak}.png',format='png')    
                
                # integrated intensity of peak
                integrated = A * np.sqrt(2*np.pi) * np.abs(sigma)    # this is wrong?
                
                print(f'\nFit of {peak} yields:')
                print('The offset of the gaussian baseline is', np.round(H,5))
                print('The center of the gaussian fit is', np.round(x0,3))
                print('The sigma of the gaussian fit is', np.round(sigma,5))
                print('The maximum intensity of the gaussian fit is', np.round(H + A,3))
                print('The Amplitude of the gaussian fit is', np.round(A,3))
                print('The FWHM of the gaussian fit is', np.round(FWHM,3))
                print('The integrated intensity is',np.round(integrated,3))   # this is wrong?
                ################################################
                
                # export integrated intensities
                peakDic[peak]['summed_counts'] = np.sum(counts)
                peakDic[peak]['peak_cut'] = [xdata,ydata]
                peakDic[peak]['monitors'] = monitors
                peakDic[peak]['fit'] = [H,x0,sigma,FWHM,integrated]
                
                if roi:
                    # plot rois
                    total = len(df.A3[startA3:stopA3])
                    rows = int(np.floor(np.sqrt(total)))
                    cols = int(np.ceil(np.sqrt(total)))
                    
                    fig,Ax = plt.subplots(nrows=rows,ncols=cols,figsize=(15,12))
                    Ax = Ax.flatten()
                    II = []
                    
                    vmin = peakDic[peak]['vmin']
                    vmax = peakDic[peak]['vmax']
                    
                    for a3,A3Idx,offset,ax in zip(df.A3[startA3:stopA3],range(startA3,stopA3), sttOffset ,Ax):
                        c = df.counts[A3Idx,startZ:stopZ,startTheta+offset:stopTheta+offset]/df.monitor[A3Idx].reshape(-1,1)
                        offsetVal = sttStepDegreesToIndex*offset
                        II.append(ax.imshow(c,origin='lower',extent=(startThetaVal+offsetVal,stopThetaVal+offsetVal,startZ,stopZ),vmin=vmin,vmax=vmax))
                        ax.set_xlabel('Two Theta [deg.]')
                        ax.set_ylabel('z [pixcel]')
                        ax.set_title(f'A3: {str(a3)}')
                        ax.axis('auto')
                    
                    fig.tight_layout()
                    
                    for i in II:
                        i.set_clim(vmin,vmax)
                    
                    if saveFig is not False:
                        fig.savefig(saveFig+f'{peak}_roi.png',format='png')   

                if closeFigures is True:
                    plt.close('all') 

        return peakDic

    def subtractBkgRange(self,bkgStart,bkgEnd,saveToFile=False, saveToNewFile = False):
        """Function generate background as defined by a range of the first dataFile of the dataSet

        Args:
            
            - bkgStart (int): start value in step for range used for background subtraction

            - bkgEnd (int): end value in step for range used for background subtraction

        Kwargs:

            - saveToFile (bool): If True, save background to data file, else save in RAM (default False)

            - saveToNewFile (string) If provided, and saveToFile is True, save a new file with the background subtraction (default False)

        """
        meanBG = self[0].counts[bkgStart:bkgEnd].mean(axis=0)/self[0].monitor[bkgStart:bkgEnd].mean(axis=0)
        for I,fg in enumerate(self):
            newBG = meanBG.reshape(128,1152)*fg.monitor[0]
            if saveToFile:
                filePath = os.path.join(fg.folder,fg.fileName)
                if saveToNewFile:
                    newNameParams = os.path.splitext(saveToNewFile)
                    newName = newNameParams[0]+'_'+str(I)+newNameParams[-1]
                    newFile = os.path.join(fg.folder,newName)
                    shutil.copyfile(filePath, newFile)
                    filePath = newFile
                    fg.fileName = newName

                with hdf.File(filePath,mode='a') as f:
                    if not f.get(HDFCountsBG) is None:
                        warnings.warn('Overwriting background in data file...')
                        del f[HDFCountsBG]
                    if not f.get(HDFTranslation['backgroundType']) is None:
                        del f[HDFTranslation['backgroundType']]
                    folder = '/'.join(HDFCountsBG.split('/')[:-1])
                    name = HDFCountsBG.split('/')[-1]
                    f[folder].create_dataset(name,data=newBG,compression=6)

                    folderType = '/'.join(HDFTranslation['backgroundType'].split('/')[:-1])
                    nameType = HDFTranslation['backgroundType'].split('/')[-1]
                    f[folderType].create_dataset(nameType,data=np.bytes_(['powder']))
            else:
                fg._background = newBG

            fg.hasBackground = True
            fg.backgroundType = 'powder'
            # or should it be multiplied with the correct monitor of the fg for all frames??? Is fg.counts divided per monitor?
            #fg._counts = counts.astype(int)
            # fg._monitor = fg.monitor[0].reshape(1,128,1152)*np.ones((fg.counts.shape[0],1,1)) # should be included to get same monitor for all a3, which we should ???
        

    def directSubtractDS(self,dsBG,saveToFile=False,saveToNewFile=False):
        """Subtracts a different dataSet one to one from the dataSet.

        Args:

            - dsBG (DataSet): dataSet that should be subtracted

        Kwargs:

            - saveToFile (bool): If True, save background to data file, else save in RAM (default False)

            - saveToNewFile (string) If provided, and saveToFile is True, save a new file with the background subtraction (default False)
            
        """
        
        for I,(fg,bg) in enumerate(zip(self,dsBG)):
            newBG = bg.counts
            if saveToFile:
                filePath = os.path.join(fg.folder,fg.fileName)
                if saveToNewFile:
                    newNameParams = os.path.splitext(saveToNewFile)
                    newName = newNameParams[0]+'_'+str(I)+newNameParams[-1]
                    newFile = os.path.join(fg.folder,newName)
                    shutil.copyfile(filePath, newFile)
                    filePath = newFile
                    fg.fileName = newName

                with hdf.File(filePath,mode='a') as f:
                    if not f.get(HDFCountsBG) is None:
                        warnings.warn('Overwriting background in data file...')
                        del f[HDFCountsBG]
                    if not f.get(HDFTranslation['backgroundType']) is None:
                        del f[HDFTranslation['backgroundType']]
                    folder = '/'.join(HDFCountsBG.split('/')[:-1])
                    name = HDFCountsBG.split('/')[-1]
                    f[folder].create_dataset(name,data=newBG,compression=6)

                    folderType = '/'.join(HDFTranslation['backgroundType'].split('/')[:-1])
                    nameType = HDFTranslation['backgroundType'].split('/')[-1]
                    f[folderType].create_dataset(nameType,data=np.bytes_(['singleCrystal']))
            else:
                fg._background = newBG

            fg.hasBackground = True
            fg.backgroundType = 'singleCrystal'
                  
    def calcualteHKLToA3A4Z(self,H,K,L,Print=True,A4Sign=-1):
        return self[0].calcualteHKLToA3A4Z(H,K,L,Print=Print,A4Sign=A4Sign)

    def export_PSI_format(self,dTheta=0.125,twoThetaOffset=0,bins=None,hourNormalization=False,outFile=None,addTitle=None,outFolder=None,useMask=False,maxAngle=5,applyCalibration=True,correctedTwoTheta=True,sampleName=True,sampleTitle=True,temperature=False,magneticField=False,electricField=False,fileNumber=False,waveLength=False):
        """
        The function takes a data set and merge the files.
        Outputs a .dat file in PSI format (Fullprof inst. 8)
        Saves the file with input name
        Data files used in the export is given in output file
        
        Kwargs:
            
            - dTheta (Float): stepsize of binning if no nins is given (default is 0.125)
            
            - twoThetaOffset (float): Linear shift of two theta, default is 0. To be used if a4 in hdf file is incorrect
            
            - Bins (list): Bins into which 2theta is to be binned (default min(2theta),max(2theta) in steps of 0.125)
            
            - outFile (str): String that will be used for outputfile. Default is automatic generated name.
            - outFolder (str): Path to folder data will be saved. Default is current working directory.
            
            - useMask (bool): export file with angular mask. Default is False
            - maxAngle (float/int): Angle of angular mask. Defualt is 5 deg. 
            
        - Arguments for automatic file name:
                
            - sampleName (bool): Include sample name in filename. Default is True.
            - sampleTitle (bool): Include sample title in filename. Default is True.
        
            - temperature (bool): Include temperature in filename. Default is False.
        
            - magneticField (bool): Include magnetic field in filename. Default is False.
        
            - electricField (bool): Include electric field in filename. Default is False.
        
            - fileNumber (bool): Include sample number in filename. Default is False.
            
            - waveLength (bool): Include waveLength in filename. Default is False. 
            
        Kwargs for sumDetector:
            - twoThetaBins (array): Actual bins used for binning (default [min(twoTheta)-dTheta/2,max(twoTheta)+dTheta/2] in steps of dTheta=0.125 Deg)
                
            - applyCalibration (bool): Use normalization files (default True)
                
            - correctedTwoTheta (bool): Use corrected two theta for 2D data (default true)
            
        Returns:
            
            .dat file in PSI format with input name
            
        Note: Input is a data set.
            
        Example:
            >>> inputNumber = _tools.fileListGenerator(565,folder)
            >>> ds = DataSet.DataSet(inputNumber)
            >>> for df in ds:
            ...    if np.any(np.isnan(df.monitor)) or np.any(np.isclose(df.monitor,0.0)):
            ...        df.monitor = np.ones_like(df.monitor)
            >>> export_PSI_format(ds)
        
        """

        twoTheta = np.asarray([[func(np.abs(df.twoTheta)) for func in [np.min,np.max]] for df in self])
        
        anglesMin = np.min(twoTheta[:,0])
        anglesMax = np.max(twoTheta[:,1])
        
        if bins is None:
            bins = np.arange(anglesMin-0.5*dTheta,anglesMax+0.51*dTheta,dTheta)

        if useMask is True:
            self.generateMask(maxAngle=maxAngle,replace=False)

        bins,intensity,err,monitor = self.sumDetector(bins,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta)
        
        bins = bins + twoThetaOffset
        
        # find mean monitor
        meanMonitor = np.median(monitor)

        # rescale intensity and err
        if hourNormalization is False:
            intensity*=meanMonitor
            err*=meanMonitor
        else:
            oneHourMonitor = (100000000)
            intensity*=oneHourMonitor
            err*=oneHourMonitor
        
        step = np.mean(np.diff(bins))
        start = bins[0]+0.5*step
        stop = bins[-1]-0.5*step
        
        temperatures = [df.temperature for df in self]
        meanTemp = np.mean(temperatures)
        stdTemp = np.std(temperatures)

        if np.all([x == self[0].sample.name for x in [s.name for s in self.sample[1:]]]):
            samName = self[0].sample.name        #.decode("utf-8")
        else:
            samName ='Unknown! Combined different sample names'
        
        if np.all([x == self[0].title for x in [s.title for s in self[1:]]]):
            samTitle = self[0].title        #.decode("utf-8")
        else:
            samTitle ='Unknown! Combined different sample titles'

        if np.all([np.isclose(df.wavelength,self[0].wavelength) for df in self[1:]]):
            wavelength = self[0].wavelength
        else:
            wavelength ='Unknown! Combined different Wavelengths'
        

        # reshape intensity and err to fit into (10,x)
        intNum = len(intensity)
        
        # How many empty values to add to allow reshape
        addEmpty = int(10*np.ceil(intNum/10.0)-intNum)
        
        intensity = np.concatenate([intensity,addEmpty*[np.nan]]).reshape(-1,10)
        err = np.concatenate([err,addEmpty*[np.nan]]).reshape(-1,10)
        
        ## Generate output to DMC file format
        titleLine = "DMC, "+samName+", "+samTitle
        if useMask is True:
            titleLine += " , anngular mask: " + str(maxAngle) + " deg." 
        paramLine = "lambda={:9.5f}, T={:8.3f}, dT={:7.3f}, Date='{}'".format(wavelength,meanTemp,stdTemp,self[0].startTime)#.decode("utf-8"))
        if hourNormalization is False:
            paramLine2= ' '+' '.join(["{:7.3f}".format(x) for x in [start,step,stop]])+" {:7.0f}".format(meanMonitor)+'., sample="'+samName+'"'
        else:
            paramLine2= ' '+' '.join(["{:7.3f}".format(x) for x in [start,step,stop]])+" {:7.0f}".format(oneHourMonitor)+'., sample="'+samName+'"'   
        dataLinesInt = '\n'.join([' '+' '.join(["{:6.0f}.".format(x).replace('nan.','    ') for x in line]) for line in intensity])
        dataLinesErr = '\n'.join([' '+' '.join(["{:7.1f}".format(x).replace('nan.','    ') for x in line]) for line in err])
        
        ## Generate bottom information part
        if len(self) == 1:
            year = 2022
            fileNumbers = str(int(self[0].fileName.split('n')[-1].split('.')[0]))
        else:
            year,fileNumbers = _tools.numberStringGenerator([df.fileName for df in self])
        
        fileList = " Filelist='dmc:{}:{}'".format(year,fileNumbers)
        
        minmax = [np.nanmin,np.nanmax]
        
        
        twoTheta = [anglesMin,anglesMax]
        Counts = [int(func(intensity)) for func in minmax]
        numor = fileNumbers.replace('-',' ')
        Npkt = len(bins) - 1        
        
        owner = self[-1].user#.decode("utf-8")
        a1 = self[-1].monochromatorRotationAngle[0]
        a2 = self[-1].monochromatorTakeoffAngle[0]
        a3 = self[-1].A3[0]
        mcv = self[-1].monochromatorCurvature[0]
        mtx = self[-1].monochromatorTranslationLower[0]
        mty = self[-1].monochromatorTranslationUpper[0]
        mgu = self[-1].monochromatorGoniometerUpper[0]
        mgl = self[-1].monochromatorGoniometerLower[0]
        
        bMon = [df.protonBeam for df in self]
        pMon = [df.monitor for df in self]
        sMon = [[0.0]]

        flattened_times = [time for df in self for time in df.time]
        timeMin, timeMax = [func(flattened_times) for func in minmax]
        sMonMin, sMonMax = [func(sMon) for func in minmax]
        bMonMin, bMonMax = [func(bMon) for func in minmax]
        aMon = np.mean([0.0 for df in self])
        pMonMin, pMonMax = [func(pMon) for func in minmax]
        muR = 0.0                           #self[-1].sample.sample_mur[0]
        preset = self[-1].mode      #.decode("utf-8")
        
        paramLines = []
        paramLines.append(" a4={:1.1f}. {:1.1f}.; Counts={} {}; Numor={}; Npkt={}; owner='{}'".format(*twoTheta,*Counts,numor,Npkt,owner))
        paramLines.append('  a1={:4.2f}; a2={:3.2f}; a3={:3.2f}; mcv={:3.2f}; mtx={:3.2f}; mty={:3.2f}; mgu={:4.3f}; mgl={:4.3f}; '.format(a1,a2,a3,mcv,mtx,mty,mgu,mgl))
        paramLines.append('  time={:4.4f} {:4.4f}; sMon={:4.0f}. {:4.0f}.; bMon={:3.0f}. {:3.0f}.; aMon={:1.0f}'.format(timeMin,timeMax,sMonMin,sMonMax,bMonMin,bMonMax,aMon))
        paramLines.append("  pMon={:7.0f}. {:7.0f}.; muR={:1.0f}.; Preset='{}'".format(float(pMonMin),float(pMonMax),muR,preset))
        paramLines.append("  calibration='{}'".format(self[-1].normalizationFile))
        paramLines.append("")
        fileString = '\n'.join([titleLine,paramLine,paramLine2,dataLinesInt,dataLinesErr,fileList,*paramLines])
        
        magneticFields = [df.magneticField for df in self]
        mag = np.mean(magneticFields)

        electricFields = [df.electricField for df in self]
        elec = np.mean(electricFields)
        
        if outFile is None:
            saveFile = "DMC"
            if hourNormalization == True:
                saveFile += "_n"
            if sampleName == True:
                saveFile += f"_{samName[:20]}"
            if sampleTitle ==True:
                saveFile += f"_{samTitle[:30]}"
            if temperature == True:
                saveFile += "_" + str(meanTemp).replace(".","p")[:5] + "K"
            if magneticField == True:
                saveFile += "_" + str(mag) + "T"
            if electricField == True:
                saveFile += "_" + str(elec) + "keV"
            if waveLength == True:
                saveFile += "_{}AA".format(str(wavelength).replace('.','p')[:5])
            if fileNumber == True:
                saveFile += "_" + fileNumbers.replace(',','_') 
            if addTitle is not None:
                saveFile += "_" + str(addTitle)
            if useMask == True:
                saveFile += '_HR'
        else:
            saveFile = str(outFile.replace('.dat',''))
            if useMask == True:
                saveFile += '_HR'
        
        saveFile=saveFile.replace('__','_').replace('__','_').replace(' ','_').replace('.','p')

        if outFolder is None:
            outFolder = os.getcwd()

        with open(os.path.join(outFolder,saveFile)+".dat",'w') as sf:
            sf.write(fileString)

    def export_xye_format(self,dTheta=0.125,twoThetaOffset=0,bins=None,hourNormalization=False,outFile=None,addTitle=None,outFolder=None,useMask=False,maxAngle=5,applyCalibration=True,correctedTwoTheta=True,sampleName=True,sampleTitle=True,temperature=False,magneticField=False,electricField=False,fileNumber=False,waveLength=False):
        """
        The function takes a data set and merge the files.
        Outputs a .xye file in with a comment line with info and xye data
        Saves the file with input name
        
        Kwargs:
            
            - dTheta (Float): stepsize of binning if no nins is given (default is 0.125)
            
            - twoThetaOffset (float): Linear shift of two theta, default is 0. To be used if a4 in hdf file is incorrect
            
            - Bins (list): Bins into which 2theta is to be binned (default min(2theta),max(2theta) in steps of 0.125)
            
            - outFile (str): String that will be used for outputfile. Default is automatic generated name.
            - outFolder (str): Path to folder data will be saved. Default is current working directory.
            - useMask (bool): export file with angular mask. Default is False
            - maxAngle (float/int): Angle of angular mask. Defualt is 5 deg. 
            
        - Arguments for automatic file name:
                
            - sampleName (bool): Include sample name in filename. Default is True.
            - sampleTitle (bool): Include sample title in filename. Default is True.
            - temperature (bool): Include temperature in filename. Default is False.
            - magneticField (bool): Include magnetic field in filename. Default is False.
            - electricField (bool): Include electric field in filename. Default is False.
            - fileNumber (bool): Include sample number in filename. Default is False.
            - waveLength (bool): Include waveLength in filename. Default is False. 
            
        Kwargs for sumDetector:
            - twoThetaBins (array): Actual bins used for binning (default [min(twoTheta)-dTheta/2,max(twoTheta)+dTheta/2] in steps of dTheta=0.125 Deg)
            - applyCalibration (bool): Use normalization files (default True)
            - correctedTwoTheta (bool): Use corrected two theta for 2D data (default true)
            
        Returns:
            
            .xye file in with a comment line with info and xye data
        
        Note: Input is a data set.
            
        Example:
            >>> inputNumber = _tools.fileListGenerator(565,folder)
            >>> ds = DataSet.DataSet(inputNumber)
            >>> for df in ds:
            ...    if np.any(np.isnan(df.monitor)) or np.any(np.isclose(df.monitor,0.0)):
            ...        df.monitor = np.ones_like(df.monitor)
            >>> export_xye_format(ds)
        """

        twoTheta = np.asarray([[func(np.abs(df.twoTheta)) for func in [np.min,np.max]] for df in self])
        
        anglesMin = np.min(twoTheta[:,0])
        anglesMax = np.max(twoTheta[:,1])
        
        if bins is None:
            bins = np.arange(anglesMin-0.5*dTheta,anglesMax+0.51*dTheta,dTheta)
        
        if useMask is True:
            self.generateMask(maxAngle=maxAngle,replace=False)

        bins,intensity,err,monitor = self.sumDetector(bins,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta)
 
        bins = bins + twoThetaOffset
        
        # find mean monitor
        meanMonitor = np.median(monitor)
        intensity[np.isnan(intensity)] = -1
        
        # rescale intensity and err
        if hourNormalization is False:
            intensity*=meanMonitor
            err*=meanMonitor
        else:
            oneHourMonitor = (100000000)
            intensity*=oneHourMonitor
            err*=oneHourMonitor
        
        step = np.mean(np.diff(bins))
        start = np.min(bins)+0.5*step
        stop = np.max(bins)-0.5*step
        
        Centres=0.5*(bins[1:]+bins[:-1])
        saveData = np.array([Centres,intensity,err])
        
        if np.all([x == self[0].sample.name for x in [s.name for s in self.sample[1:]]]):
            samName = self[0].sample.name        #.decode("utf-8")
        else:
            samName ='Unknown! Combined different sample names'
        
        if np.all([x == self[0].title for x in [s.title for s in self[1:]]]):
            samTitle = self[0].title        #.decode("utf-8")
        else:
            samTitle ='Unknown! Combined different sample titles'

        if np.all([np.isclose(df.wavelength,self[0].wavelength) for df in self[1:]]):
            wavelength = self[0].wavelength
        else:
            wavelength ='Unknown! Combined different Wavelengths'
            
        
        temperatures = np.array([df.temperature for df in self])
        meanTemp = np.mean(temperatures)
        
        magneticFields = [df.magneticField for df in self]
        mag = np.mean(magneticFields)

        electricFields = [df.electricField for df in self]
        elec = np.mean(electricFields)

        if len(self) == 1:
            year = 2022
            fileNumbers = str(int(self[0].fileName.split('n')[-1].split('.')[0]))
        else:
            year,fileNumbers = _tools.numberStringGenerator([df.fileName for df in self])
        
        titleLine1 = f"# DMC at SINQ, PSI: Sample name = {samName}, title = {samTitle}, wavelength = {str(wavelength)[:5]} AA, T = {str(meanTemp)[:5]} K"
        titleLine2 = "# Filelist='dmc:{}:{}'".format(year,fileNumbers)
        if useMask is True:
            titleLine2 += " , anngular mask: " + str(maxAngle) + " deg." 
        if hourNormalization is False:
            titleLine3= '# '+' '.join(["{:7.3f}".format(x) for x in [start,step,stop]])+" {:7.0f}".format(meanMonitor)+', sample="'+samName+'"'
        else:
            titleLine3= '# '+' '.join(["{:7.3f}".format(x) for x in [start,step,stop]])+" {:7.0f}".format(oneHourMonitor)+', sample="'+samName+'"'
       
        if outFile is None:
            saveFile = "DMC"
            if hourNormalization == True:
                saveFile += "_n"
            if sampleName == True:
                saveFile += f"_{samName[:20]}"
            if sampleTitle ==True:
                saveFile += f"_{samTitle[:30]}"
            if temperature == True:
                saveFile += "_" + str(meanTemp).replace(".","p")[:5] + "K"
            if magneticField == True:
                saveFile += "_" + str(mag) + "T"
            if electricField == True:
                saveFile += "_" + str(elec) + "keV"
            if waveLength == True:
                saveFile += "_{}AA".format(str(wavelength).replace('.','p')[:5])
            if fileNumber == True:
                saveFile += "_" + fileNumbers.replace(',','_') 
            if addTitle is not None:
                saveFile += "_" + str(addTitle)
            if useMask == True:
                saveFile += '_HR'
        else:
            saveFile = str(outFile.replace('.xye',''))
            if useMask == True:
                saveFile += '_HR'

        saveFile=saveFile.replace('__','_').replace('__','_').replace(' ','_').replace('.','p')

        if outFolder is None:
            outFolder = os.getcwd()

        with open(os.path.join(outFolder,saveFile)+".xye",'w') as sf:
            sf.write(titleLine1+"\n")    
            sf.write(titleLine2+"\n") 
            sf.write(titleLine3+"\n") 
            np.savetxt(sf,saveData.T,delimiter='  ')
            sf.close()
         

    def updateDataFiles(self,key,value):
        """Update a property across all data files
        
        Args:
            - key (str): parameter name
            - value (float): parameter value
        """
        if np.all([hasattr(df,key) for df in self]): # all datafiles have the key
            try:
                length = len(value)
            except TypeError:
                length = 1
            
            if length == len(self): # input has the same length as number of data files! Apply individually
                if length == 1:
                    value = [value]
                for v,df in zip(value,self):
                    setattr(df,key,v)
            elif length == 1:
                for df in self:
                    setattr(df,key,value)
            else:
                raise AttributeError('Length of DataSet is {} but received {} values for {}'.format(len(self),length,key))
                
            self._getData() # update!
            
        else:
            missing = len(self)-np.sum([hasattr(df,key) for df in self])
            if missing == 0:
                raise AttributeError('DataFiles do not contain',key)
            else:
                raise AttributeError('Not all DataFiles do not contain',key)

    def cutQPlane(self,points, width, dQx = None, dQy = None, xBins =None, yBins =None, rlu=False, steps=None, sample = None):
        """Perform QPlane cut where points within +-0.5*width are collapsed onto the plane and binned into xBins and yBins
        Args:
            - points (list): List of three points within the wanted plane. X is parallel to point 2 - point 1 (p1, p2, p3 = points)

            - width (float): Total width of QPlane in units of 1/AA or rlu depending on the rlu flag
        
        Kwargs:
        
            - dQx (float): Step size along x if xBins is not provided (default None)
            
            - dQy (float): Step size along y if yBins is not provided (default None)
            
            - xBins (list): Binning edges along x, overwrites dQx (default None)
            
            - yBins (list): Binning edges along y, overwrites dQy (default None)
            
            - rlu (bool): If true utilize sample UB otherwise perform no rotation (default False)
            
            - steps (int): Number of a3 step computated at once when performing operation (default len(df))

            - sample (Sample): Use specified sample for RLU axis if RLU = True (default None = self.sample[0])
        
        If dQx and dQy is set an automatic binning size is performed, however an error will be thrown if neither dQx (dQy) and  xBins (yBins) are set.

        """
        if np.all([x is None for x in [dQx,dQy,xBins,yBins]]):
            raise AttributeError('No bins or step sizes provided')

        if sample is None:
            sample = self[0].sample

        if xBins is None:
            if dQx is None:
                raise AttributeError('Neither dQx or xBins are set!')
        if yBins is None:
            if dQy is None:
                raise AttributeError('Neither dQx or xBins are set!')

       
        if yBins is None and xBins is None:
            bins = None # Automatic binning
            autoBins = True
        else: # One of the bins is set
            autoBins = False
            if yBins is None:
                yBins = np.arange(-5,5,dQy)
            elif xBins is None:
                xBins = np.arange(-5,5,dQx)


        if not points is None:
            if rlu:
                newPoints = [np.dot(sample.UB,point) for point in points]
            else:
                newPoints = points
            
            totalRotMat,translation = _tools.calculateRotationMatrixAndOffset2(newPoints)
        else:
            totalRotMat = np.eye(3)
            translation = np.asarray([0.0])
        
        returndata = None
        for df in self:
            
            if steps is None:
                steps = len(df)
            
            stepsTaken = 0
            
            totalRotMatDF = totalRotMat
            
            for idx in _tools.arange(0,len(df),steps):
                
                q = np.einsum('ij,jk->ik',totalRotMatDF,df.q[idx[0]:idx[-1]].reshape(3,-1),optimize='greedy')
                mask = df.mask[idx[0]:idx[-1]]
                
                # Check that the points are in the plane and take only the local x and y coordinates
                inside = np.logical_and(np.abs(q[2]-translation)<width*0.5,np.logical_not(mask.flatten()))
                q = q[:2,inside]
                print(df.fileName,'from',idx[0],'to',idx[-1])
                if q.shape[1] == 0:
                    print('Empty slices. Continuing...')
                    continue
                if autoBins:
                    
                    xMin,xMax = q[0].min(), q[0].max()
                    yMin,yMax = q[1].min(), q[1].max()
                    if bins is None: # initial calculation of bins
                        xBins = np.arange(xMin-0.51*dQx,xMax+0.51*dQx,dQx)
                        yBins = np.arange(yMin-0.51*dQy,yMax+0.51*dQy,dQy)
                        bins = (xBins,yBins)
                    else:
                        
                        if xMin < xBins[0] or xMax > xBins[-1]:
                            lowExtensionX = np.max([int(np.ceil((bins[0][0]-xMin)/dQx)),0])
                            highExtensionX = np.max([int(np.ceil((xMax-bins[0][-1])/dQx+0.1)),0])

                            xBins = np.arange(bins[0][0]-lowExtensionX*dQx,bins[0][-1]+highExtensionX*dQx+0.4*dQx,dQx)
                            bins = (xBins,bins[1])
                        else:
                            lowExtensionX = highExtensionX = 0
                        if yMin < yBins[0] or yMax > yBins[-1]:
                            lowExtensionY = np.max([int(np.ceil((bins[1][0]-yMin)/dQy)),0])
                            highExtensionY = np.max([int(np.ceil((yMax-bins[1][-1])/dQy+0.1)),0])

                            yBins = np.arange(bins[1][0]-lowExtensionY*dQy,bins[1][-1]+highExtensionY*dQy+0.4*dQy,dQy)
                            bins = (bins[0],yBins)
                        else:
                            lowExtensionY = highExtensionY = 0    
                            
                            # if any extension is nonzero, rescale
                        if np.any(np.asarray([lowExtensionX,lowExtensionY,highExtensionX,highExtensionY])!=0):
                            dat = []
                            for mat in returndata:

                                tempMat = np.zeros((len(bins[0])-1,len(bins[1])-1),dtype=mat.dtype)
                                tempMat[lowExtensionX:lowExtensionX+mat.shape[0],lowExtensionY:lowExtensionY+mat.shape[1]] = mat
                                mat = tempMat
                                dat.append(mat)
                            returndata = dat          
                
                dat = df.intensitySliced(slice(idx[0],idx[1]))
                    
                mon = df.monitor[idx[0]:idx[1]]
                mon=np.repeat(np.repeat(mon[:,np.newaxis],dat.shape[1],axis=1)[:,:,np.newaxis],dat.shape[2],axis=-1)
                
                stepsTaken+=steps

                I = df.counts[idx[0]:idx[1]]
                Norm = df.normalization#[idx[0]:idx[1]]
                Norm = np.repeat(Norm[np.newaxis],len(I),axis=0).flatten()[inside]
                
                I = I.flatten()[inside]
                dat = dat.flatten()[inside]
                mon = mon.flatten()[inside]
                weights = [I,mon,Norm]
                
                intensity,monitorCount,Normalization,NormCount = _tools.histogramdd(q.T,bins=(xBins,yBins),weights=weights,returnCounts=True)

                if returndata is None:
                    returndata = [intensity,monitorCount,Normalization,NormCount]
                else:
                    returndata = [rd+x for rd,x in zip(returndata,[intensity,monitorCount,Normalization,NormCount])]

        Qx =np.outer(xBins,np.ones_like(yBins))
        Qy =np.outer(np.ones_like(xBins),yBins)
        bins = [Qx,Qy]

        return returndata,bins,totalRotMat,translation

    def plotQPlane(self,points, width, sample=None, dQx = None, dQy = None, xBins =None, yBins =None, rlu=False, steps=None,log=False,ax=None,rmcFile=False,**kwargs):
        """Wrapper for plotting tool to show binned intensities in the Q plane between provided Qz values.

        Args:
            - points (list): List of three points within the wanted plane. X is parallel to point 2 - point 1 (p1, p2, p3 = points)
            - width (float): Total width of QPlane in units of 1/AA or rlu depending on the rlu flag
        
        Kwargs:
        
            - dQx (float): Step size along x if xBins is not provided (default None)
            - dQy (float): Step size along y if yBins is not provided (default None)
            - xBins (list): Binning edges along x, overwrites dQx (default None)
            - yBins (list): Binning edges along y, overwrites dQy (default None)
            - rlu (bool): If true utilize sample UB otherwise perform no rotation (default False)
            - steps (int): Number of a3 step computated at once when performing operation (default len(df))
            - sample (Sample): Use specified sample for RLU axis if RLU = True (default None = self.sample[0])
            - log (bool): Plot intensities as the logarithm (default False).
            - ax (matplotlib axes): Axes in which the data is plotted (default None). If None, the function creates a new axes object.
            - rlu (bool): If true and axis is None, a new reciprocal lattice axis is created and used for plotting (default True).
            - vmin (float): Lower limit for colorbar (default min(Intensity)).
            - vmax (float): Upper limit for colorbar (default max(Intensity)).
            - colorbar (bool): If True, a colorbar is created in figure (default False)
            - zorder (int): If provided decides the z ordering of plot (default 10)
            - other: Other key word arguments are passed to the pcolormesh plotting algorithm.
            
        Returns:
            
            - dataList (list): List of all data points in format [Intensity, Monitor, Normalization, Normcount]
            - bins (list): List of bin edges as function of plane in format [xBins,yBins].
            - ax (matplotlib axes): Returns provided matplotlib axis
            
        .. note::
            The axes object has a new method denoted 'set_clim' taking two parameters (VMin and VMax) used to change axes colouring.
            The axes object has a new method denoted 'to_csv' taking one parameter, fileName, which is where the csv is saved.

        """
        
        if 'zorder' in kwargs:
            zorder = kwargs['zorder']
            kwargs = _tools.without_keys(dictionary=kwargs,keys='zorder')
        else:
            zorder = 10

        if 'cmap' in kwargs:
            cmap = kwargs['cmap']
            kwargs = _tools.without_keys(dictionary=kwargs,keys='cmap')
        else:
            cmap = None

        returndata,bins,rotationMatrix,translation = self.cutQPlane(points=points,sample=sample,width=width,dQx=dQx,dQy=dQy,xBins=xBins,yBins=yBins,rlu=rlu,steps=steps)

        if sample is None:
            sample = copy.deepcopy(self[0].sample)

        if ax is None:
            if rlu:
                s = copy.deepcopy(self[0].sample)
                p1 = points[1]-points[0]
                p2 = points[2]-points[0]

                p1Q = np.dot(s.B,p1)
                p2Q = np.dot(s.B,p2)

                p3 = _tools.LengthOrder(np.dot(np.linalg.inv(s.B),np.cross(p1Q,p2Q)))

                s.P1 = p1
                s.P2 = p2
                s.P3 = p3
                s.projectionVectors = np.array([s.P1,s.P2,s.P3]).T
                s.ROT = rotationMatrix


                ax = self.createRLUAxes(sample=s)
                ax._step = translation
    
            else:
                fig,ax = plt.subplots()
                ax.set_xlabel('Qx [1/AA]')
                ax.set_ylabel('Qy [1/AA]')    
        
        ax.intensity,ax.monitorCount,ax.Normalization,ax.NormCount, = returndata

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.Int = np.divide(ax.intensity*ax.NormCount,ax.monitorCount*ax.Normalization)
        ax.bins = bins
        ax.Qx,ax.Qy = ax.bins
        
        if log:
            ax.Int = np.log10(1e-20+np.array(ax.Int))
        else:
            ax.Int = np.asarray(ax.Int)

        if 'vmin' in kwargs:
            vmin = kwargs['vmin']
            kwargs = _tools.without_keys(dictionary=kwargs,keys='vmin')
        else:
            vmin = np.nanmin([np.nanmin(intens) for intens in ax.Int])

        if 'vmax' in kwargs:
            vmax = kwargs['vmax']
            kwargs = _tools.without_keys(dictionary=kwargs,keys='vmax')
        else:
            vmax = np.nanmax([np.nanmax(intens) for intens in ax.Int])

        if 'colorbar' in kwargs:
            colorbar = kwargs['colorbar']
            kwargs = _tools.without_keys(dictionary=kwargs,keys='colorbar')
        else:
            colorbar = False
        pmeshs = []
        
        ax.grid(False)
        pmeshs.append(ax.pcolormesh(ax.Qx,ax.Qy,ax.Int,zorder=zorder,cmap=cmap,**kwargs))
        ax.set_aspect('equal')
        ax.grid(True, zorder=0)
        
        if 'pmeshs' in ax.__dict__:
            ax.pmeshs = np.concatenate([ax.pmeshs,np.asarray(pmeshs)],axis=0)
        else:
            ax.pmeshs = pmeshs


        def set_clim(pmeshs,vmin,vmax):
            for pmesh in pmeshs:
                pmesh.set_clim(vmin,vmax)
        ax.set_clim = lambda vMin,vMax: set_clim(ax.pmeshs,vMin,vMax)



        if colorbar:
            ax.colorbar = ax.get_figure().colorbar(ax.pmeshs[0],pad=0.1)
            ax.colorbar.set_label('$I$ [arb.u.]', rotation=270)

        ax.set_clim(vmin,vmax)
        
        ax.QzMean = translation
            
        if len(ax.Qx)!=0:
            xmin = np.min([np.min(qx) for qx in ax.Qx])
            xmax = np.max([np.max(qx) for qx in ax.Qx])
            ax.set_xlim(xmin,xmax)#np.min(Qx),np.max(Qx))
        
        if len(ax.Qy)!=0:
            ymin = np.min([np.min(qy) for qy in ax.Qy])
            ymax = np.max([np.max(qy) for qy in ax.Qy])
            ax.set_ylim(ymin,ymax)#np.min(Qy),np.max(Qy))
    
        
        def to_csv(fileName,ax,rmcFile,rmcFileName):
            Qx,Qy = ax.bins
            QxCenter = 0.25*(Qx[:-1,:-1]+Qx[:-1,1:]+Qx[1:,1:]+Qx[1:,:-1])
            QyCenter = 0.25*(Qy[:-1,:-1]+Qy[:-1,1:]+Qy[1:,1:]+Qy[1:,:-1])
            QzCenter = np.full(Qx[:-1,:-1].shape,ax.QzMean)

            QxQyQz_AX = np.asarray([QxCenter,QyCenter,QzCenter]) # size [3,Nx,Ny]
            inversRot = np.linalg.inv(ax.sample.ROT)
            QxQyQz_DF = np.einsum('ij,j...->i...',inversRot,QxQyQz_AX)

            H,K,L = ax.sample.calculateQxQyQzToHKL(*QxQyQz_DF)
            intensity,monitorCount,Normalization,NormCount = ax.data
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Int = np.divide(intensity*NormCount,Normalization*monitorCount)
                Int[np.isnan(Int)] = -1
                Int_err = np.divide(np.sqrt(intensity)*NormCount,Normalization*monitorCount)
                Int_err[np.isnan(Int_err)] = -1
            dataToPandas = {'Qx':QxCenter.flatten(),'Qy':QyCenter.flatten(),'Qz':QzCenter.flatten(),'H':H.flatten(),'K':K.flatten(),'L':L.flatten(), 'Intensity':intensity.flatten(), 'Monitor':monitorCount.flatten(),
                            'Normalization':Normalization.flatten(),'BinCounts':NormCount.flatten(),'Int':Int.flatten(),'Int_err':Int_err.flatten()}
            ax.d = pd.DataFrame(dataToPandas)

            with open(fileName,'w') as f:
                f.write("# CSV generated from DMCpy {}. Shape of data is {}\n".format(DMCpy.__version__,Int.shape))

            ax.d.to_csv(fileName,mode='a')

            if rmcFile is True:
                dataToRMC = {'H':H.flatten(),'K':K.flatten(),'L':L.flatten(),'Int':Int.flatten(),'Int_err':Int_err.flatten()}
                ax.e = pd.DataFrame(dataToRMC)
                if rmcFileName is None:
                    rmcFileName = 'sample_xtal_data_01.txt'
                ax.e.to_csv(rmcFileName, header=None, index=None, sep=' ', mode='w')

        
        #ax.to_csv = lambda fileName : to_csv(fileName,ax,spinteract,rmcFileName)
        if rmcFile is True:
            ax.to_csv = lambda fileName, rmcFileName : to_csv(fileName, ax, True, rmcFileName)
        else:
            ax.to_csv = lambda fileName : to_csv(fileName,ax,False,None)

        ax.data = [ax.intensity,ax.monitorCount,ax.Normalization,ax.NormCount]
        return ax,returndata,bins

    def setProjectionVectors(self,p1,p2,p3=None):
        """Set or update the projection vectors used for the View3D
        
        Args:

            - p1 (list): New primary projection, in HKL

            - p2 (list): New secondary projection, in HKL

        Kwargs:

            - p3 (list): New tertiary projection, in HKL. If None, orthogonal to p1 and p2 (default None)
        """
        for sample in self.sample:
            sample.setProjectionVectors(p1=p1,p2=p2,p3=p3)

            
def add(*listinput,PSI=True,xye=False,folder=None,outFolder=None,dataYear=None,dTheta=0.125,twoThetaOffset=0,bins=None,outFile=None,addTitle=None,useMask=True,onlyHR=False,maxAngle=5,hourNormalization=True,onlyNorm=True,applyCalibration=True,correctedTwoTheta=True,sampleName=True,sampleTitle=True,temperature=False,magneticField=False,electricField=False,fileNumber=False,waveLength=False):

    """
    
    Takes a set/series file numbers and export a added/merged file. 
    The input is read as a tuple and can be formatted as int, str, list, and several arguments separated by comma can be given. 
    If one argument is a list or str, multiple filenumbers can be given inside.
    
    Exports PSI and xye format file for all scans. 
    
    Kwargs:
        
        - listinput (tuple): File numbers for files that should be added. Examples: 1234, 1235, 1236-1238. All files will be merged.
        
        - folder (str): Path to directory for data files, default is current working directory
        
        - PSI (bool): Export PSI format. Default is True
        
        - xye (bool): Export xye format. Default is True
        - outFolder (str): Path to folder data will be saved. Default is current working directory.
        
        - all from export_PSI_format and export_xye_format
        - useMask (bool): export file with angular mask. Default is True
        - maxAngle (float/int): Angle of angular mask. Defualt is 5 deg. 
        
        - hourNormalization (bool): export files normalized to one hour on monitor.
        - onlyHR (bool): export only data with an angular mask
        - onlyNorm (bool): export only data normalized to one hour files
        - dataYear (int): year of data collection
        
    - Arguments for automatic file name:
            
        - sampleName (bool): Include sample name in filename. Default is True.
        - sampleTitle (bool): Include sample title in filename. Default is True.
    
        - temperature (bool): Include temperature in filename. Default is False.
    
        - magneticField (bool): Include magnetic field in filename. Default is False.
    
        - electricField (bool): Include electric field in filename. Default is False.
    
        - fileNumber (bool): Include sample number in filename. Default is False.
        - waveLength (bool): Include waveLength in filename. Default is False. 
        - addTitle (str): for adding text in addition to the automatically generated file name
        
    Kwargs for sumDetector:
        - twoThetaBins (array): Actual bins used for binning (default [min(twoTheta)-dTheta/2,max(twoTheta)+dTheta/2] in steps of dTheta=0.125 Deg)
            
        - applyCalibration (bool): Use normalization files (default True)
            
        - correctedTwoTheta (bool): Use corrected two theta for 2D data (default true)
        
    Example:    
        >>> add(565,566,567,(570),'571-573',[574],sampleName=False,temperature=False)
        
        output:
            DMC_565-567_570-574 as both .dat and xye files
    """    

    if folder is None:
        folder = os.getcwd()
    if outFolder is None:
        outFolder = os.getcwd()
        
    listOfDataFiles = str()
    
    if type(listinput) == tuple:
        for elemnt in listinput:
            elemnt = str(elemnt)
            elemnt = elemnt.replace('"','').replace("'","").replace('(','').replace(')','').replace('[','').replace(']','').strip(',')
            listOfDataFiles += f"{elemnt},"
        print(f"Export of added files: {listOfDataFiles[:-1]}")
        inputNumber = _tools.fileListGenerator(listOfDataFiles[:-1],folder=folder,year=dataYear)
        ds = DataSet(inputNumber)
        try:
            if PSI is True and onlyHR is False:
                if onlyNorm is False:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            if xye is True and onlyHR is False:
                if onlyNorm is False:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            if PSI is True and useMask is True:
                if onlyNorm is False:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)        
            if xye is True and useMask is True:
                if onlyNorm is False:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
        except:
                print(f"Cannot export! File is wrong format: {elemnt}")                    


def export(*listinput,PSI=True,xye=False,folder=None,outFolder=None,dataYear=None,dTheta=0.125,twoThetaOffset=0,bins=None,outFile=None,addTitle=None,useMask=True,onlyHR=False,maxAngle=5,hourNormalization=True,onlyNorm=True,applyCalibration=True,correctedTwoTheta=True,sampleName=True,sampleTitle=True,temperature=False,magneticField=False,electricField=False,fileNumber=False,waveLength=False):
    """
    Takes a set file numbers and export induvidually. 
    The input is read as a tuple and can be formatted as int, str, list, and arguments separated by comma is export induvidually. 
    If one argument is a list or str, multiple filenumbers can be given inside, and they will be added/merged.
    
    Exports PSI and xye format file for all scans. 
    
     Kwargs:
        
        - listinput (tuple): File numbers for files that should be exported. Examples: 1234, 1235, 1236-1238. File numbers separated by comma will be exported induvidually.
        
        - folder (str): Path to directory for data files, default is current working directory
        
        - PSI (bool): Export PSI format. Default is True
        
        - xye (bool): Export xye format. Default is True
        - outFolder (str): Path to folder data will be saved. Default is current working directory.
        
        - all from export_PSI_format and export_xye_format
        - useMask (bool): export file with angular mask. Default is True
        - maxAngle (float/int): Angle of angular mask. Defualt is 5 deg. 
        
        - hourNormalization (bool): export files normalized to one hour on monitor.
        - onlyHR (bool): export only data with an angular mask
        - onlyNorm (bool): export only data normalized to one hour files
        - dataYear (int): year of data collection
        
    - Arguments for automatic file name:
            
        - sampleName (bool): Include sample name in filename. Default is True.
        - sampleTitle (bool): Include sample title in filename. Default is True.
    
        - temperature (bool): Include temperature in filename. Default is False.
    
        - magneticField (bool): Include magnetic field in filename. Default is False.
    
        - electricField (bool): Include electric field in filename. Default is False.
    
        - fileNumber (bool): Include sample number in filename. Default is False.
        - waveLength (bool): Include waveLength in filename. Default is False. 
        - addTitle (str): for adding text in addition to the automatically generated file name
        
    Kwargs for sumDetector:
        - twoThetaBins (array): Actual bins used for binning (default [min(twoTheta)-dTheta/2,max(twoTheta)+dTheta/2] in steps of dTheta=0.125 Deg)
            
        - applyCalibration (bool): Use normalization files (default True)
            
        - correctedTwoTheta (bool): Use corrected two theta for 2D data (default true)
        
    Example:    
        >>> export(565,'566',[567,568,570,571],'570-573',(574,575),sampleName=None,temperature=False)  
        
        output: DMC_565, DMC_566, DMC_567_568_570_571, DMC_570-573, DMC_574-575 as both .dat and xye files
        
    """    
    if folder is None:
        folder = os.getcwd()
    if outFolder is None:
        outFolder = os.getcwd()

    for elemnt in listinput:
        elemnt = str(elemnt)
        elemnt = elemnt.replace('"','').replace("'","").replace('(','').replace(')','').replace('[','').replace(']','').strip(',')
        print(f"Export of: {elemnt}")
        inputNumber = _tools.fileListGenerator(elemnt,folder,year=dataYear)
        ds = DataSet(inputNumber)
        try:
            if PSI is True and onlyHR is False:
                if onlyNorm is False:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            if xye is True and onlyHR is False:
                if onlyNorm is False:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            if PSI is True and useMask is True:
                if onlyNorm is False:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)        
            if xye is True and useMask is True:
                if onlyNorm is False:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
        except:
                print(f"Cannot export! File is wrong format: {elemnt}")                    


def exportAll(*listinput,PSI=True,xye=False,folder=None,outFolder=None,dataYear=None,dTheta=0.125,twoThetaOffset=0,bins=None,outFile=None,addTitle=None,useMask=True,onlyHR=False,maxAngle=5,hourNormalization=True,onlyNorm=True,applyCalibration=True,correctedTwoTheta=True,sampleName=True,sampleTitle=True,temperature=False,magneticField=False,electricField=False,fileNumber=False,waveLength=False):
    """
    Takes a set file numbers and export induvidually. 
    The input is read as a tuple and can be formatted as int, str, list, and arguments separated by comma is export induvidually. 
    If one argument is a list or str, multiple filenumbers can be given inside, and they will be added/merged.
    
    Exports PSI and xye format file for all scans.  
    
    Kwargs:
        
        - listinput (tuple): the function will export all elements of the tuple/list inducidually. Files can be merged by [], '', and () notation.
        
        - folder (str): Path to directory for data files, default is current working directory
        
        - PSI (bool): Export PSI format. Default is True
        
        - xye (bool): Export xye format. Default is True
        - outFolder (str): Path to folder data will be saved. Default is current working directory.
        
        - all from export_PSI_format and export_xye_format
        - useMask (bool): export file with angular mask. Default is True
        - maxAngle (float/int): Angle of angular mask. Defualt is 5 deg. 
        
        - hourNormalization (bool): export files normalized to one hour on monitor.
        - onlyHR (bool): export only data with an angular mask
        - onlyNorm (bool): export only data normalized to one hour files
        - dataYear (int): year of data collection
        
    - Arguments for automatic file name:
            
        - sampleName (bool): Include sample name in filename. Default is True.
        - sampleTitle (bool): Include sample title in filename. Default is True.
    
        - temperature (bool): Include temperature in filename. Default is False.
    
        - magneticField (bool): Include magnetic field in filename. Default is False.
    
        - electricField (bool): Include electric field in filename. Default is False.
    
        - fileNumber (bool): Include sample number in filename. Default is False.
        - waveLength (bool): Include waveLength in filename. Default is False. 
        - addTitle (str): for adding text in addition to the automatically generated file name
        
    Kwargs for sumDetector:
        - twoThetaBins (array): Actual bins used for binning (default [min(twoTheta)-dTheta/2,max(twoTheta)+dTheta/2] in steps of dTheta=0.125 Deg)
            
        - applyCalibration (bool): Use normalization files (default True)
            
        - correctedTwoTheta (bool): Use corrected two theta for 2D data (default true)
        
    Example:    
        >>> export(565,'566',[567,568,570,571],'570-573',(574,575),sampleName=None,temperature=False)  
        
        output: DMC_565, DMC_566, DMC_567_568_570_571, DMC_570-573, DMC_574-575 as both .dat and xye files
        
    """    

    if folder is None:
        folder = os.getcwd()
    if outFolder is None:
        outFolder = os.getcwd()
    
    if type(listinput) in [tuple,list,str]:
        listinput = str(listinput)
        listinput = listinput.replace('"','').replace("'","").replace('(','').replace(')','').replace('[','').replace(']','').strip(',')
        inputNumbers = _tools.fileListGenerator(listinput,folder,year=dataYear)
        for elemnt in inputNumbers:
            dataYear, fileNumbers = _tools.numberStringGenerator([elemnt])
            print(f"Export of: {fileNumbers}")
            ds = DataSet([elemnt])
            try:
                if PSI is True and onlyHR is False:
                    if onlyNorm is False:
                        ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                    if hourNormalization is True:
                        ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if xye is True and onlyHR is False:
                    if onlyNorm is False:
                        ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                    if hourNormalization is True:
                        ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if PSI is True and useMask is True:
                    if onlyNorm is False:
                        ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                    if hourNormalization is True:
                        ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)        
                if xye is True and useMask is True:
                    if onlyNorm is False:
                        ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                    if hourNormalization is True:
                        ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            except:
                    print(f"Cannot export! File is wrong format: {elemnt}")                    

def export_from(startFile,PSI=True,xye=False,folder=None,outFolder=None,dataYear=None,dTheta=0.125,twoThetaOffset=0,bins=None,outFile=None,addTitle=None,useMask=True,onlyHR=False,maxAngle=5,hourNormalization=True,onlyNorm=True,applyCalibration=True,correctedTwoTheta=True,sampleName=True,sampleTitle=True,temperature=False,magneticField=False,electricField=False,fileNumber=False,waveLength=False):

    """
    
    Takes a starting file number and export xye format file for all the following files in the folder.
    Exports PSI and xye format file for all scans. 
    
    Kwargs:
        
        - startFile (int): First file number for export
        
        - endFile (int): Final file number for export
        
        - folder (str): Path to directory for data files, default is current working directory
        
        - PSI (bool): Export PSI format. Default is True
        
        - xye (bool): Export xye format. Default is True
        - outFolder (str): Path to folder data will be saved. Default is current working directory.
        
        - all from export_PSI_format and export_xye_format
        - useMask (bool): export file with angular mask. Default is True
        - maxAngle (float/int): Angle of angular mask. Defualt is 5 deg. 
        
        - hourNormalization (bool): export files normalized to one hour on monitor.
        - onlyHR (bool): export only data with an angular mask
        - onlyNorm (bool): export only data normalized to one hour files
        - dataYear (int): year of data collection
        
    - Arguments for automatic file name:
            
        - sampleName (bool): Include sample name in filename. Default is True.
        - sampleTitle (bool): Include sample title in filename. Default is True.
    
        - temperature (bool): Include temperature in filename. Default is False.
    
        - magneticField (bool): Include magnetic field in filename. Default is False.
    
        - electricField (bool): Include electric field in filename. Default is False.
    
        - fileNumber (bool): Include sample number in filename. Default is False.
        - waveLength (bool): Include waveLength in filename. Default is False. 
        - addTitle (str): for adding text in addition to the automatically generated file name
        
    Kwargs for sumDetector:
        - twoThetaBins (array): Actual bins used for binning (default [min(twoTheta)-dTheta/2,max(twoTheta)+dTheta/2] in steps of dTheta=0.125 Deg)
            
        - applyCalibration (bool): Use normalization files (default True)
            
        - correctedTwoTheta (bool): Use corrected two theta for 2D data (default true)
        
    Example:       
        >>> export_from(590,sampleName=False,temperature=False)    
        
    """
    if folder is None:
        folder = os.getcwd()
    if outFolder is None:
        outFolder = os.getcwd()

    hdf_files = [f for f in os.listdir(folder) if f.endswith('.hdf')]
    last_hdf = hdf_files[-1]

    numberOfFiles = int(last_hdf.strip('.hdf').split('n')[-1]) - int(startFile)
    
    fileList = list(range(startFile,startFile+numberOfFiles))
    
    for file in fileList:  

        file = str(file)
        file = file.replace('"','').replace("'","").replace('(','').replace(')','').replace('[','').replace(']','').replace(' ','').strip(',')
        print(f"Export of: {file}")
        inputNumber = _tools.fileListGenerator(file,folder,dataYear)
        ds = DataSet(inputNumber)
        try:
            if PSI is True and onlyHR is False:
                if onlyNorm is False:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            if xye is True and onlyHR is False:
                if onlyNorm is False:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            if PSI is True and useMask is True:
                if onlyNorm is False:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)        
            if xye is True and useMask is True:
                if onlyNorm is False:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
        except:
                print(f"Cannot export! File is wrong format: {file}")                    


def export_from_to(startFile,endFile,PSI=True,xye=False,folder=None,outFolder=None,dataYear=None,dTheta=0.125,twoThetaOffset=0,bins=None,outFile=None,addTitle=None,useMask=True,onlyHR=False,maxAngle=5,hourNormalization=True,onlyNorm=True,applyCalibration=True,correctedTwoTheta=True,sampleName=True,sampleTitle=True,temperature=False,magneticField=False,electricField=False,fileNumber=False,waveLength=False):

    """
    
    Takes a starting file number and a end file number, export for all scans between (including start and end)
    Exports PSI and xye format file for all scans. 
    
    Kwargs:
        
        - startFile (int): First file number for export
        
        - endFile (int): Final file number for export
        
        - folder (str): Path to directory for data files, default is current working directory
        
        - PSI (bool): Export PSI format. Default is True
        
        - xye (bool): Export xye format. Default is True
        - outFolder (str): Path to folder data will be saved. Default is current working directory.
        
        - all from export_PSI_format and export_xye_format
        - useMask (bool): export file with angular mask. Default is True
        - maxAngle (float/int): Angle of angular mask. Defualt is 5 deg. 
        
        - hourNormalization (bool): export files normalized to one hour on monitor.
        - onlyHR (bool): export only data with an angular mask
        - onlyNorm (bool): export only data normalized to one hour files
        - dataYear (int): year of data collection
        
    - Arguments for automatic file name:
            
        - sampleName (bool): Include sample name in filename. Default is True.
        - sampleTitle (bool): Include sample title in filename. Default is True.
    
        - temperature (bool): Include temperature in filename. Default is False.
    
        - magneticField (bool): Include magnetic field in filename. Default is False.
    
        - electricField (bool): Include electric field in filename. Default is False.
    
        - fileNumber (bool): Include sample number in filename. Default is False.
        - waveLength (bool): Include waveLength in filename. Default is False. 
        - addTitle (str): for adding text in addition to the automatically generated file name
        
    Kwargs for sumDetector:
        - twoThetaBins (array): Actual bins used for binning (default [min(twoTheta)-dTheta/2,max(twoTheta)+dTheta/2] in steps of dTheta=0.125 Deg)
            
        - applyCalibration (bool): Use normalization files (default True)
            
        - correctedTwoTheta (bool): Use corrected two theta for 2D data (default true)
        
    Example:     
        >>> export_from_to(565,570,sampleName=False,temperature=False)
        
        output: DMC_565, DMC_566, DMC_567, DMC_568, DMC_569, DMC__570 as both .xye and .dat files
        
    """
    if folder is None:
        folder = os.getcwd()
    if outFolder is None:
        outFolder = os.getcwd()

    fileList = list(range(startFile,endFile+1))
    
    for file in fileList:    
        file = str(file)
        file = file.replace('"','').replace("'","").replace('(','').replace(')','').replace('[','').replace(']','').replace(' ','').strip(',')
        print(f"Export of: {file}")
        inputNumber = _tools.fileListGenerator(file,folder,dataYear)
        ds = DataSet(inputNumber)
        try:
            if PSI is True and onlyHR is False:
                if onlyNorm is False:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            if xye is True and onlyHR is False:
                if onlyNorm is False:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            if PSI is True and useMask is True:
                if onlyNorm is False:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)        
            if xye is True and useMask is True:
                if onlyNorm is False:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
        except:
                print(f"Cannot export! File is wrong format: {file}")                    



def export_list(listinput,PSI=True,xye=False,folder=None,outFolder=None,dataYear=None,dTheta=0.125,twoThetaOffset=0,bins=None,outFile=None,addTitle=None,useMask=True,onlyHR=False,maxAngle=5,hourNormalization=True,onlyNorm=True,applyCalibration=True,correctedTwoTheta=True,sampleName=True,sampleTitle=True,temperature=False,magneticField=False,electricField=False,fileNumber=False,waveLength=False):
    """
    Takes a list and export all elements induvidually. If a list is given inside the list, these files will be added/merged.
    Exports PSI and xye format file for all scans. 
    
    Kwargs:
        
        - list input (list: List of files that will be exported.
        
        - folder (str): Path to directory for data files, default is current working directory
        
        - PSI (bool): Export PSI format. Default is True
        
        - xye (bool): Export xye format. Default is True
        - outFolder (str): Path to folder data will be saved. Default is current working directory.
        
        - all from export_PSI_format and export_xye_format
        - useMask (bool): export file with angular mask. Default is True
        - maxAngle (float/int): Angle of angular mask. Defualt is 5 deg. 
        
        - hourNormalization (bool): export files normalized to one hour on monitor.
        - onlyHR (bool): export only data with an angular mask
        - onlyNorm (bool): export only data normalized to one hour files
        - dataYear (int): year of data collection
        
    - Arguments for automatic file name:
            
        - sampleName (bool): Include sample name in filename. Default is True.
        - sampleTitle (bool): Include sample title in filename. Default is True.
    
        - temperature (bool): Include temperature in filename. Default is False.
    
        - magneticField (bool): Include magnetic field in filename. Default is False.
    
        - electricField (bool): Include electric field in filename. Default is False.
    
        - fileNumber (bool): Include sample number in filename. Default is False.
        - waveLength (bool): Include waveLength in filename. Default is False. 
        - addTitle (str): for adding text in addition to the automatically generated file name
        
    Kwargs for sumDetector:
        - twoThetaBins (array): Actual bins used for binning (default [min(twoTheta)-dTheta/2,max(twoTheta)+dTheta/2] in steps of dTheta=0.125 Deg)
            
        - applyCalibration (bool): Use normalization files (default True)
            
        - correctedTwoTheta (bool): Use corrected two theta for 2D data (default true)
        
    Example:    
        >>> export_list([565,566,567,[569,570]],sampleName=False,temperature=False) 
        
        output: DMC_565, DMC_566, DMC_567, DMC_569_570 as both .xye and .dat files
        
    """    

    if folder is None:
        folder = os.getcwd()
    if outFolder is None:
        outFolder = os.getcwd()
        
    for file in listinput:   
        file = str(file)
        file = file.replace('"','').replace("'","").replace('(','').replace(')','').replace('[','').replace(']','').replace(' ','').strip(',')
        print(f"Export of: {file}")           
        inputNumber = _tools.fileListGenerator(file,folder,dataYear)
        ds = DataSet(inputNumber)
        try:
            if PSI is True and onlyHR is False:
                if onlyNorm is False:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            if xye is True and onlyHR is False:
                if onlyNorm is False:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=False,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
            if PSI is True and useMask is True:
                if onlyNorm is False:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_PSI_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)        
            if xye is True and useMask is True:
                if onlyNorm is False:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=False,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
                if hourNormalization is True:
                    ds.export_xye_format(dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,addTitle=addTitle,outFolder=outFolder,useMask=useMask,maxAngle=maxAngle,hourNormalization=hourNormalization,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,sampleTitle=sampleTitle,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber,waveLength=waveLength)    
        except:
                print(f"Cannot export! File is wrong format: {file}")                    

                
def subtract_PSI(file1,file2,outFile=None,folder=None,outFolder=None):
    """
    This function takes two .dat files in PSI format and export a differnce curve with correct uncertainties. 
    
    The second file is scaled after the monitor of the first file.
    
    Kwargs:
        
        - PSI (bool): Subtract PSI format. Default is True
        
        - xye (bool): Subtract xye format. Default is True
        - folder (str): Path to directory for data files, default is current working directory
        
        - outFile (str): string for name of outfile (given without extension)
        - outFolder (str): Path to folder data will be saved. Default is current working directory.
                
    Example:
        >>> subtract('DMC_565.dat','DMC_573')
    
    """

    if folder is None:
        folder = os.getcwd()
        
    with open(os.path.join(folder,file1.replace('.dat','')+'.dat'),'r') as rf:
        allinfo1 = rf.readlines()
        rf.close()

    with open(os.path.join(folder,file2.replace('.dat','')+'.dat'),'r') as rf:
        allinfo2 = rf.readlines()
        rf.close()    

    info1 = allinfo1[:3] 
    info2 = allinfo2[:3] 

    if info1[2].split(',')[0].split(',')[0] != info2[2].split(',')[0].split(',')[0]:
        return print('Not same range of files! Cannot subtract.')          
        
    infoStr1 = (info1[2].split(',')[0].strip('#').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' '))
    infoArr1 = [float(x) for x in infoStr1[1:].split(' ')]

    infoStr2 = (info2[2].split(',')[0].strip('#').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' '))
    infoArr2 = [float(x) for x in infoStr2[1:].split(' ')]
        
    monitor1 = infoArr1[3]
    monitor2 = infoArr2[3]
    monitorRatio = monitor1/monitor2    
    dataPoints = int((infoArr1[2]-infoArr1[0]) / infoArr1[1]) + 1
    subInt = []
    subErr = []
    
    dataLines = int(np.ceil(dataPoints/10)) 
    commentlines = 3
    
    for intLines in range(dataLines): 
        subIntList = []
        subErrList = []
        intline1= allinfo1[intLines+commentlines]
        intline2= allinfo2[intLines+commentlines]
        errline1= allinfo1[intLines+dataLines+commentlines]
        errline2= allinfo2[intLines+dataLines+commentlines]
        intensity1 = [float(x) for x in intline1[:-2].replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('nan','').replace('na','').split(' ') if x != '' ]  
        intensity2 = [float(x) for x in intline2[:-2].replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('nan','').replace('na','').split(' ') if x != '' ] 
        err1 = [float(x) for x in errline1[:-2].replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('nan','').replace('na','').split(' ') if x != '' ] 
        err2 = [float(x) for x in errline2[:-2].replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('nan','').replace('na','').split(' ') if x != '' ] 
        for i, j in zip(intensity1,intensity2):
            subIntList.append(i-j*monitorRatio)
        for h, k in zip(err1,err2):
            subErrList.append(np.sqrt(h**2 + monitorRatio**2 * k**2))
        subInt.append(subIntList)
        subErr.append(subErrList)
    
    titleLine = str(info1[0]).strip('\n') + ', subtracted: ' + str(info2[0])  + str(info1[1]) + str(info1[2]).strip('\n')
    dataLinesInt = '\n'.join([' '+' '.join(["{:6.0f}.".format(x) for x in line]) for line in subInt])
    dataLinesErr = '\n'.join([' '+' '.join(["{:7.1f}".format(x) for x in line]) for line in subErr])
    indexParamLines = dataLines*2 + commentlines
    paramLine1 = '\n'.join([str(line).strip('\n') for line in allinfo1[indexParamLines:]])
    paramLine2 = ' subtracted:'
    paramLine3 = str(info2[0])  + str(info1[1]) + str(info1[2]).strip('\n')
    paramLine4 = ''.join([str(line) for line in allinfo2[indexParamLines:]])
    fileString = '\n'.join([titleLine,dataLinesInt,dataLinesErr,paramLine1,paramLine2,paramLine3,paramLine4])
    
    if outFile is None:
        saveFile = file1.replace('.dat','') + '_sub_' + file2.replace('.dat','')
    else:
        saveFile = str(outFile.replace('.dat',''))

    print(f'Subtracting PSI: {file1}.dat minus {file2}.dat') 

    if outFolder is None:
        outFolder = os.getcwd()

    with open(os.path.join(outFolder,saveFile)+".dat",'w') as sf:
        sf.write(fileString)

def subtract_xye(file1,file2,outFile=None,folder=None,outFolder=None):
    """
    This function takes two .xye files and export a differnce curve with correct uncertainties. 
    
    The second file is scaled after the monitor of the first file.
    
    Kwargs:
        
        - PSI (bool): Subtract PSI format. Default is True
        
        - xye (bool): Subtract xye format. Default is True
        - folder (str): Path to directory for data files, default is current working directory
        
        - outFile (str): string for name of outfile (given without extension)
        - outFolder (str): Path to folder data will be saved. Default is current working directory. 
        
    Example:
        >>> subtract('DMC_565.xye','DMC_573')
        
    """
    
    if folder is None:
        folder = os.getcwd()
    
    data1 = np.genfromtxt(os.path.join(folder,file1.replace('.xye','')+'.xye'), delimiter='  ')
    data2 = np.genfromtxt(os.path.join(folder,file2.replace('.xye','')+'.xye'), delimiter='  ')  
    
    with open(os.path.join(folder,file1.replace('.xye','')+'.xye'),'r') as rf:
        info1 = rf.readlines()[:3]
        rf.close()

    with open(os.path.join(folder,file2.replace('.xye','')+'.xye'),'r') as rf:
        info2 = rf.readlines()[:3]
        rf.close()

    if info1[2].split(',')[0].split(',')[0] != info2[2].split(',')[0].split(',')[0]:
        return print('Not same range of files! Cannot subtract.')          
        
    infoStr1 = (info1[2].split(',')[0].strip('#').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' '))
    infoArr1 = [float(x) for x in infoStr1[1:].split(' ')]

    infoStr2 = (info2[2].split(',')[0].strip('#').replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' '))
    infoArr2 = [float(x) for x in infoStr2[1:].split(' ')]

    monitorRatio = infoArr1[3]/infoArr2[3]

    subInt = np.subtract(data1[:,1], np.multiply(monitorRatio,(data2[:,1])))

    intErr2 = monitorRatio * data2[:,2]
    
    subErr = 0 * data2[:,1]
    
    for i in range(len(data1[:,2])):
        subErr[i] = np.sqrt( (data1[i,2])**2 + (intErr2[i])**2 ) 
    
    saveData = np.array([data1[:,0],subInt,subErr])

    if outFile is None:
        saveFile = file1.replace('.xye','') + '_sub_' + file2.replace('.xye','')
    else:
        saveFile = str(outFile.replace('.xye',''))

    print(f'Subtracting xye: {file1}.xye minus {file2}.xye')    

    if outFolder is None:
        outFolder = os.getcwd()

    with open(os.path.join(outFolder,saveFile)+".xye",'w') as sf:
        sf.write('# ' + str(info1) + "\n")   
        sf.write("# subtracted file: \n") 
        sf.write('# ' + str(info2) + "\n") 
        np.savetxt(sf,saveData.T,delimiter='  ')
        sf.close()

def subtract(file1,file2,PSI=True,xye=True,outFile=None,folder=None,outFolder=None):
    """
    This function takes two files and export a differnce curve with correct uncertainties. 
    The second file is scaled after the monitor of the first file.
    Kwargs:
        
        - PSI (bool): Subtract PSI format. Default is True
        
        - xye (bool): Subtract xye format. Default is True
        
        - folder (str): Path to directory for data files, default is current working directory
        
        - outFile (str): string for name of outfile (given without extension)
        - outFolder (str): Path to folder data will be saved. Default is current working directory. 
        
    Example:
        >>> subtract('DMC_565.xye','DMC_573')
    """


    if folder is None:
        folder = os.getcwd()

    file1 = file1.replace('.xye','').replace('.dat','')
    file2 = file2.replace('.xye','').replace('.dat','')

    if PSI == True:
        try:
            subtract_PSI(file1,file2,outFile,folder=folder,outFolder=outFolder)
        except:
            print('Cannot subtract PSI format files')
    if xye == True:
        try:
            subtract_xye(file1,file2,outFile,folder=folder,outFolder=outFolder)
        except:
            print('Cannot subtract xye format files')
        
def DMCsort(filelist,sortKey):
    """Sort a list of files based on a sortKey
    Args:
        - filelist (list): List of data file locations to sort. 
        - sortKey (str): sorting key

    return:
        sorted list
    """
    names =  shallowRead(filelist,[str(sortKey)])
    
    listOfFiles = []
    listOfTitles = []

    for name in names:
        listOfFiles.append(name['file'])
        listOfTitles.append(name[sortKey])
        
    sumFile = {}

    for file,title in zip(listOfFiles,listOfTitles):
        if title in sumFile:
            pass
        else:
            sumFile[title] = [file]

    return sumFile


def sortExport(fileList,dataFolder=None,PSI=True,xye=True,outFolder=None,dTheta=0.125,twoThetaOffset=0,bins=None,outFile=None,applyCalibration=True,correctedTwoTheta=True,sampleName=True,temperature=False,magneticField=False,electricField=False,fileNumber=False):
              
    localSettingsFile = os.path.join(os.environ['LNSG_HOME'],'DMCpySettings.json')
    if not os.path.isfile(localSettingsFile):
        print('Cannot find local settings file (',localSettingsFile,')')
        
    else:
        with open(localSettingsFile) as f:
            experiment = json.load(f)
            if dataFolder is None:
                dataFolder = '/afs/psi.ch/project/sinqdata/{0}/dmc/{1}'.format(experiment['year'],experiment['proposalNumber'])
                # dataFolder = r'C:\Users\fjellv_o\switchdrive\DMC\test/{0}/{1}'.format(experiment['year'],experiment['proposalNumber'])
            
    sampleSort = DMCsort(_tools.fileListGenerator(fileList,dataFolder),'sampleName')
    
    sampleTitleSort = {}
    
    for key in sampleSort.keys():
        if key == '':
            sampleTitleSort = DMCsort(sampleSort[key],'title')
        else:
            sampleTitleSort = DMCsort(sampleSort[key],'title')
        for key in sampleTitleSort.keys():
            year, fileNumbers = _tools.numberStringGenerator(sampleTitleSort[key])
            DataSet.add(fileNumbers,folder=dataFolder,PSI=PSI,xye=xye,outFolder=outFolder,dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber)
  

def sortExportLong(fileListLong,dataFolder=None,PSI=True,xye=True,outFolder=None,dTheta=0.125,twoThetaOffset=0,bins=None,outFile=None,applyCalibration=True,correctedTwoTheta=True,sampleName=True,temperature=False,magneticField=False,electricField=False,fileNumber=False):
              
    if dataFolder is None:
        dataFolder = os.getcwd()    
    
    sampleSort = DMCsort(fileListLong,'sampleName')
    
    sampleTitleSort = {}
    
    for key in sampleSort.keys():
        if key == '':
            pass
        else:
            sampleTitleSort = DMCsort(sampleSort[key],'title')
        for key in sampleTitleSort.keys():
            year, fileNumbers = _tools.numberStringGenerator(sampleTitleSort[key])
            DataSet.add(fileNumbers,folder=dataFolder,PSI=PSI,xye=xye,outFolder=outFolder,dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber)
 

def listGenerator(start=None,end=None):
        
    localSettingsFile = os.path.join(os.environ['LNSG_HOME'],'DMCpySettings.json')
    if not os.path.isfile(localSettingsFile):
        print('Cannot find local settings file (',localSettingsFile,')')
        
    else:
        with open(localSettingsFile) as f:
            experiment = json.load(f)
            dataFolder = '/afs/psi.ch/project/sinqdata/{0}/dmc/{1}'.format(experiment['year'],experiment['proposalNumber'])
            # dataFolder = r'C:\Users\fjellv_o\switchdrive\DMC\test/{0}/{1}'.format(experiment['year'],experiment['proposalNumber'])
            
    hdf_files = [f for f in os.listdir(dataFolder) if f.endswith('.hdf')]
    
    if start is None:
        start = int(hdf_files[0].strip('.hdf').split('n')[-1])
    if end is None:
        end = int(hdf_files[-1].strip('.hdf').split('n')[-1])
        
    print('generating sorted list for: ', start, ' to ', end)
    
    fileList = list(range(start,end+1))
    fileList = str(fileList)
    fileList = fileList.replace('"','').replace("'","").replace('(','').replace(')','').replace('[','').replace(']','').replace(' ','').strip(',')
    
    fileListLong = _tools.fileListGenerator(fileList,dataFolder)
    
    return fileList, fileListLong, dataFolder


def sleepExport(sleep_time,start=None,end=None,PSI=True,xye=True,outFolder=None,dTheta=0.125,twoThetaOffset=0,bins=None,outFile=None,applyCalibration=True,correctedTwoTheta=True,sampleName=True,temperature=False,magneticField=False,electricField=False,fileNumber=False):
          
    Flag = True
    while Flag:
        fileList, fileListLong, dataFolder = listGenerator(start=start,end=end)
        sortExportLong(fileListLong,dataFolder,PSI=PSI,xye=xye,outFolder=outFolder,dTheta=dTheta,twoThetaOffset=twoThetaOffset,bins=bins,outFile=outFile,applyCalibration=applyCalibration,correctedTwoTheta=correctedTwoTheta,sampleName=sampleName,temperature=temperature,magneticField=magneticField,electricField=electricField,fileNumber=fileNumber)
        print(f'waiting {sleep_time} s')
        time.sleep(float(sleep_time)) 


def export_help(): 
    print(" ")
    print(" The following commands are avaliable for export of powder data in DMCpy:")
    print(" ")
    print("     export, add, export_from, export_from_to, export_list")
    print(" ")
    print(" They export both PSI and xye format by default. Can be deactivated by the arguments PSI=False and xye=False")
    print(" ")
    print('      - export: For export of induvidual sets of scan files. Files can be merged by [] or "" notation, i.e. list or strings.')
    print('      - add: TThe function adds/merge all the files given. ')
    print("      - export_from: For export of all data files in a folder after a startfile")
    print("      - export_from_to: It exports all files between and including two given files")
    print("      - export_list: Takes a list and export all the files separatly. If a list is given inside the list, the files will be merged ")
    print(" ")
    print(" Examples:  ")
    print('      >>> export(565,"566",[567,568,570,571],"570-573",(574,575),sampleName=False,temperature=False)  ')
    print("      >>> add(565,566,567,(570),'571-573',[574],sampleName=False,temperature=False) ")
    print("      >>> export_from(590,fileNumber=True)  ")
    print("      >>> export_from_to(565,570,dTheta=0.25,twoThetaOffset=2.0)")
    print("      >>> export_list([565,566,567,570],temperature=False,xye=False)")
    print("      >>> export_list([565,566,567,[568,569,570]]) # This is an example of list inside a list. 568,569,570 will be merged in this case. ")
    print('      >>> add("565,567,570-573",outFile="mergefilename")')
    print('      >>> export(565,folder=r"Path\To\Data\Folder")   #Note r"..." notation')      
    print(" ")
    print(" Most important kewords and aguments:")
    print("     - dTheta (float): stepsize of binning if no bins is given (default is 0.125)")
    print("     - outFile (str): String that will be used for outputfile. Default is automatic generated name.")
    print("     - outFolder (str): Path to folder data will be saved. Default is current working directory.")
    print("     - twoThetaOffset (float): Linear shift of two theta, default is 0. To be used if a4 in hdf file is incorrect")
    print(" ")
    print(" Arguments for automatic file name:")
    print("     - sampleName (bool): Include sample name in filename. Default is True.")
    print("     - temperature (bool): Include temperature in filename. Default is True.")
    print("     - fileNumber (bool): Include sample number in filename. Default is True.")
    print("     - magneticField (bool): Include magnetic field in filename. Default is False.")
    print("     - electricField (bool): Include electric field in filename. Default is False.")
    print(" ")
    print(" There is also a subtract function for subtracting PSI format files and xye format files. ")    
    print(" The files are normalized to the onitor of the first dataset.")
    print(" Input is two existing filenames with or without extenstion. ")
    print(" PSI and xye format can be deactivated by PSI = False and xye = False")    
    print(" Alternatively can subtract_PSI or subtract_xye be used")
    print(" ")
    print("      >>> subtract('DMC_565.xye','DMC_573')")    
    print(" ")
    print(" ")
    print(" ")
    print(" ")

from matplotlib.ticker import FuncFormatter

def generate1DAxis(q1,q2,rlu=True,outputFunction=print):
    fig,ax = plt.subplots()
    ax = plt.gca()
    q1 = np.asarray(q1,dtype=float)
    q2 = np.asarray(q2,dtype=float)
    
    if rlu:
        variables = ['H','K','L']
    else:
        variables = ['Qx','Qy','Qz']
    
        # Start points defined form cut
    ax.startPoint = q1
    ax.endPoint = q2
    
    
    # plot direction is q2-q1, without normalization making all x points between 0 and 1
    
    ax.plotDirection = _tools.LengthOrder(np.array(q2-q1)).reshape(-1,1)
    
    # Calculate the needed precision for x-axis plot
    def calculateXPrecision(ax):
        # Find diff for current view
        diffPlotPosition = np.diff(ax.get_xlim())[0]
        diffAlongPlot = ax.plotDirection*diffPlotPosition
        
        numTicks = len(ax.xaxis.get_ticklocs())
        
        # take the smallest value which is chaning (i.e. is along the plot direction)
        minChange = np.min(np.abs(diffAlongPlot[ax.plotDirection.T.flatten()!=0])) /numTicks 
        
        
        # find the largest integer closest to the wanted precision
        ax.set_precision(int(-np.floor(np.log10(minChange)))+1)
    

    def calculateIndex(binDistance,x):
        idx = np.argmin(np.abs(binDistance-x))
        return idx
    
    def calculatePosition(ax,x):
        if isinstance(x,(np.ndarray)):
            return (x*ax.plotDirection.T+ax.startPoint)
        else:
            return (x*ax.plotDirection.T+ax.startPoint).flatten()
    
    def calculatePositionInv(ax,h,k,l):
        HKL = np.asarray([h,k,l])
        return np.dot((HKL-ax.startPoint.reshape(3,1)).T,ax.plotDirection)/(np.dot(ax.plotDirection.T,ax.plotDirection)).T
        # return np.dot((HKL-ax.startPoint.reshape(3,1)).T,ax.plotDirection.reshape(3,1))/(np.dot(ax.plotDirection.T,ax.plotDirection))
    
    # Add methods to the axis
    
    

    ax._x_precision = 2
    ax.fmtPrecisionString = '{:.'+str(2)+'f}'
    # Dynamic add setter and getter to ax.precision
    
    def set_precision(ax,value):
        ax._x_precision = value
        ax.fmtPrecisionString = '{:.'+str(ax._x_precision)+'f}'
        ax.get_figure().tight_layout()
        
    
    
    ax.calculatePosition = lambda x: calculatePosition(ax,x)
    ax.calculatePositionInv = lambda h,k,l: calculatePositionInv(ax,h,k,l)
    
    ax.calculateIndex = lambda x: calculateIndex(ax.Data['binDistance'],x)
    ax.calculateXPrecision = calculateXPrecision
    ax.set_precision = lambda value: set_precision(ax,value)
    ax.calculateXPrecision(ax)
    
    # Format the x label as well as the format_coord
    if rlu==False:
        xlabel = r'[$Q_x [\AA^{-1}]$, $Q_y [\AA^{-1}]$, $Q_z [\AA^{-1}]$]'
        
        ax.set_xlabel(xlabel)
        def format_coord(x,y,ax):# pragma: no cover
            qx,qy,qz = ax.calculatePosition(x)
            return  "qx = {0:.3e}, qy = {1:.3e}, qz = {2:.3e}, I = {3:0.4e}".format(qx,qy,qz,y)
    else:
        xlabel = '[$Q_h$ [RLU], $Q_k$ [RLU], $Q_l$ [RLU]]'
        ax.set_xlabel(xlabel)
        
        def format_coord(x,y,ax):# pragma: no cover
            h,k,l = ax.calculatePosition(x)
            return  "H = {0:.3e}, K = {1:.3e}, L = {2:.3e}, I = {3:0.4e}".format(h,k,l,y)
        
    
    # Create a custom major formatter to show the multi-D position on the x-axis
    def major_formatter(ax,tickPosition,tickNumber):
        positions = list(ax.calculatePosition(tickPosition))
        return '\n'.join([ax.fmtPrecisionString.format(pos) for pos in positions])
    
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,i: major_formatter(ax,x,i)))
    
    # Create the onclick behaviour
    def onclick(event,ax,outputFunction):# pragma: no cover
        if ax.in_axes(event):
            try:
                C = ax.get_figure().canvas.cursor().shape() # Only works for pyQt5 backend
            except:
                pass
            else:
                if C != 0: # Cursor corresponds to arrow
                    return
    
            x = event.xdata
            y = event.ydata
            printString = ax.format_coord(x,y)
            
            outputFunction(printString)
    
    
    # connect methods
    ax.format_coord = lambda x,y: format_coord(x,y,ax)
    ax._button_press_event = ax.figure.canvas.mpl_connect('button_press_event',lambda event:onclick(event,ax,outputFunction=outputFunction))
    
    ax.callbacks.connect('xlim_changed',ax.calculateXPrecision)
    # Make the layouyt fit
    ax.get_figure().tight_layout()

    return ax
