# SPDX-License-Identifier: MPL-2.0
import functools
import sys
sys.path.append('.')
import numpy as np
from difflib import SequenceMatcher
import os.path
import cProfile, pstats, io
from itertools import product
import pickle
import h5py as hdf
import datetime, shutil
from DMCpy.FileStructure import shallowRead, HDFTranslationAlternatives, HDFTranslation, HDFCounts

import DMCpy


MPLKwargs = ['agg_filter','alpha','animated','antialiased','aa','clip_box','clip_on','clip_path','color','c','colorbar','contains','dash_capstyle','dash_joinstyle','dashes','drawstyle','figure','fillstyle','gid','label','linestyle or ls','linewidth or lw','marker','markeredgecolor or mec','markeredgewidth or mew','markerfacecolor or mfc','markerfacecoloralt or mfcalt','markersize or ms','markevery','path_effects','picker','pickradius','rasterized','sketch_params','snap','solid_capstyle','solid_joinstyle','transform','url','visible','xdata','ydata','zorder']

def KwargChecker(function=None,include=None):
    """Function to check if given key-word is in the list of accepted Kwargs. If not directly therein, checks capitalization. If still not match raises error
    with suggestion of closest argument.
    
    Args:
    
        - func (function): Function to be decorated.

    Raises:

        - AttributeError
    """
    def KwargCheckerNone(func):
        @functools.wraps(func)
        def newFunc(*args,**kwargs):
            argList = extractArgsList(func,newFunc,function,include)
            checkArgumentList(argList,kwargs)
            returnval = func(*args,**kwargs)
            return returnval
        newFunc._original = func
        newFunc._include = include
        newFunc._function = function
        return newFunc
    return KwargCheckerNone

def extractArgsList(func,newFunc,function,include):
    N = func.__code__.co_argcount # Number of arguments with which the function is called
    argList = list(newFunc._original.__code__.co_varnames[:N]) # List of arguments
    if not function is None:
        if isinstance(function,(list,np.ndarray)): # allow function kwarg to be list or ndarray
            for f in function:
                for arg in f.__code__.co_varnames[:f.__code__.co_argcount]: # extract all arguments from function
                    argList.append(str(arg))
        else: # if single function
            for arg in function.__code__.co_varnames[:function.__code__.co_argcount]:
                argList.append(str(arg))
    if not include is None:
        if isinstance(include,(list,np.ndarray)):
            for arg in include:
                argList.append(str(arg))
        else:
            argList.append(str(include))
        argList = list(set(argList)) # Cast to set to remove duplicates
        argList.sort() #  Sort alphabetically
    return argList

def checkArgumentList(argList,kwargs):
    notFound = []
    for key in kwargs:
        if key not in argList:
            similarity = np.array([SequenceMatcher(None, key.lower(), x.lower()).ratio() for x in argList])
            maxVal = np.max(similarity)
            maxId = np.argmax(similarity)
            notFound.append('Key-word argument "{}" not understood. Did you mean "{}"?'.format(key,argList[maxId]))
    if len(notFound)>0:
        if len(notFound)>1:
            errorMsg = 'The following key-word arguments are not understood:\n'
            errorMsg+='\n'.join(notFound)
        else:
            errorMsg = notFound[0]
        error = AttributeError(errorMsg)
        raise error

@KwargChecker()
def numberStringGenerator(fileNames,instrumentName='dmc'):
    names = np.array([os.path.splitext(os.path.basename(df))[0] for df in fileNames])
    # Find base name and remove extension
    if len(fileNames) != 1:
        prefix = os.path.commonprefix(list(names))

        if instrumentName in prefix:
            # Remove all non-zero digits from prefix
            while prefix[-1]!='n': #  prefix[-1]!='0' and
                prefix = prefix[:-1]
            year = int(prefix[len(instrumentName):len(instrumentName)+4])
            numbers = np.array([n[len(prefix):] for n in names],dtype=int)
            sortNumbers = np.sort(numbers)
            diff = np.diff(sortNumbers)
            separators = list(np.arange(len(diff))[diff>1]+1) # add one due to diff removing 1 lenght
            groups = []
            if len(separators) == 0:
                groups.append('-'.join([str(sortNumbers[0]),str(sortNumbers[-1])]))
            else:
                separators.insert(0,0)
                separators.append(-1)
                for start,stop in zip(separators[:-1],separators[1:]):
                    if stop == -1:
                        group = sortNumbers[start:]
                    else:
                        group = sortNumbers[start:stop]
                    if len(group)>2:
                        groups.append('-'.join([str(group[0]),str(group[-1])]))
                    elif len(group)==2:
                        groups.append(','.join(group.astype(str)))
                    else:
                        groups.append(str(group[0]))
            files = ','.join(groups)
    else:
        splitting = fileNames[0].split('n')
        InstrumentYear = 'n'.join(splitting[:-1])
        fileNumbers = splitting[-1]
        files = str(int(fileNumbers.split('.')[0]))
        year = int(InstrumentYear[-4:])
    return year,files

@KwargChecker()
def fileListGenerator(numberString,folder,year=2021, format = None, instrument = 'dmc'):
    """Function to generate list of data files.
    
    Args:
        
        - numberString (str): List if numbers separated with comma and dashes for sequences.
        
        - folder (str): Folder of wanted data files.
        
    Kwargs:

        - year (int): Year of wanted data files (default 2018)

        - format (str): format of data files (default None, but dmc if instrument is provided)

        - instrument (str): Instrument to be used to determine format string (default dmc)
        
    returns:
        
        - list of strings: List containing the full file string for each number provided.
        
    Example:
        >>> numberString = '201-205,207-208,210,212'
        >>> files = fileListGenerator(numberString,'data/',2018)
        ['data/dmc2018n000201.hdf', 'data/dmc2018n000202.hdf', 
        'data/dmc2018n000203.hdf', 'data/dmc2018n000204.hdf', 
        'data/dmc2018n000205.hdf', 'data/dmc2018n000207.hdf', 
        'data/dmc2018n000208.hdf', 'data/dmc2018n000210.hdf', 
        'data/dmc2018n000212.hdf']
    """
        
    splits = numberString.split(',')
    dataFiles = []
    if format is None: # If no user specified format is provided
        if instrument == 'dmc':
            format = 'dmc{:d}n{:06d}.hdf'
        else:
            raise AttributeError('Provided instrument "{}" not understood'.format(instrument))


    for sp in splits:
        isRange = sp.find('-')!=-1
        
        if isRange:
            spSplits = sp.split('-')
            if len(spSplits)>2:
                raise AttributeError('Sequence "{}" not understood - too many dashes.'.format(sp))
            startNumber = int(spSplits[0])
            endNumber = int(spSplits[1])
            numbers = np.arange(startNumber,endNumber+1)    
        else:
            numbers = [int(sp)]

        dataFiles.append([os.path.join(folder,format.format(year,x)) for x in numbers])
    return list(np.concatenate(dataFiles))

def roundPower(x,default=4):
    """Round to nearest 10^x"""
    if not np.isclose(x,0.0): # Sign to fit with np.round
        return -int(np.floor(np.log10(np.abs(x))))
        
    else:
        return default


def binData3D(dx,dy,dz,pos,data,norm=None,mon=None,bins=None):
    """ 3D binning of data.

    Args:

        - dx (float): Step size in x (required).

        - dy (float): Step size in x (required).

        - dz (float): Step size in x (required).

        - pos (2D array): Position of data points as flattened lists (X,Y,Z) (required).

        - data (array): Flattened data array (required).

    Kwargs:

        - norm (array): Flattened normalization array.

        - mon (array): Flattened monitor array.

        - bins (list of arrays): Bins locating edges in the x, y, and z directions.

    returns:

        Re-binned intensity (and if provided Normalization, Monitor, and Normalization Count) and X, Y, and Z bins in 3 3D arrays.


    Example:

    >>> pos = [Qx,Qy,E]
    >>> Data,bins = DataSet.binData3D(0.05,0.05,0.2,pos,I,norm=Norm,mon=Monitor)

    """

    if bins is None:
        bins = calculateBins(dx=dx,dy=dy,dz=dz,pos=pos)
    if len(pos[0].shape)>1: # Flatten positions
        pos = np.array([x.flatten() for x in pos])
    #NonNaNs = 1-np.isnan(data.flatten())

    #pos = [np.array(x[NonNaNs]) for x in pos]
    HistBins = [bins[0][:,0,0],bins[1][0,:,0],bins[2][0,0,:]]

    if False:
        intensity =    np.histogramdd(np.array(pos).T,bins=HistBins,weights=data.flatten())[0].astype(data.dtype)

        returndata = [intensity]
        if mon is not None:
            MonitorCount=  np.histogramdd(np.array(pos).T,bins=HistBins,weights=mon.flatten())[0].astype(mon.dtype)
            returndata.append(MonitorCount)
        if norm is not None:
            Normalization= np.histogramdd(np.array(pos).T,bins=HistBins,weights=norm.flatten())[0].astype(norm.dtype)
            
            returndata.append(Normalization)
            
        NormCount =    np.histogramdd(np.array(pos).T,bins=HistBins,weights=np.ones_like(data).flatten())[0].astype(float)
        returndata.append(NormCount)

    else:
        weights = [data.flatten()]
        if not mon is None:
            weights.append(mon.flatten())
        if not norm is None:
            weights.append(norm.flatten())
        
        returndata = histogramdd(np.array(pos).T,bins=HistBins,weights=weights,returnCounts=True)

    return returndata,bins


def calculateBins(dx,dy,dz,pos):
    diffx = np.abs(np.max(pos[0])-np.min(pos[0]))
    diffy = np.abs(np.max(pos[1])-np.min(pos[1]))
    diffz = np.abs(np.max(pos[2])-np.min(pos[2]))
    
    xbins = np.round(diffx/dx).astype(int)+1
    ybins = np.round(diffy/dy).astype(int)+1
    zbins = np.round(diffz/dz).astype(int)+1
    
    _X = np.linspace(np.min(pos[0]),np.max(pos[0]),xbins)
    _Y = np.linspace(np.min(pos[1]),np.max(pos[1]),ybins)
    _Z = np.linspace(np.min(pos[2]),np.max(pos[2]),zbins)
    
    X,Y,Z = np.meshgrid(_X,_Y,_Z,indexing='ij')
    
    XX,YY,ZZ = calculateGrid3D(X=X,Y=Y,Z=Z)
    
    bins=[XX,YY,ZZ]
    return bins



def calculateGrid3D(X,Y,Z):
    """Generate 3D grid with centers given by X,Y, and Z.
     Args:
        
        X (3D array): 3D array of x values generated by np.meshgrid.
                
        Y (3D array): 3D array of y values generated by np.meshgrid.
                
        Z (3D array): 3D array of z values generated by np.meshgrid.
        
    Example:

    >>> x = np.linspace(-1.5,1.5,20)
    >>> y = np.linspace(0,1.5,10)
    >>> z = np.linspace(-1.0,5.5,66)
    >>> X,Y,Z = np.meshgrid(x,y,z,indexing='ij')
    >>> XX,YY,ZZ = calculateGrid3D(X,Y,Z)

    Now XX is a 21x11x67 array containing all x coordinates of the edges exactly midway between the points. Same goes for YY and ZZ with y and z coordinates respectively.
    """

    xshape = np.array(X.shape)
    if np.any(xshape <= 1):
        raise AttributeError('Provided array has dimension(s) {} of size <= 1'.format(xshape))
    XT = np.zeros((xshape[0]+1,xshape[1]+1,xshape[2]+1))
    YT = np.zeros_like(XT)
    ZT = np.zeros_like(XT)
    
    
    
    dx0 = np.diff(X,axis=0)
    dx1 = np.diff(X,axis=1)
    dx2 = np.diff(X,axis=2)
    dy0 = np.diff(Y,axis=0)
    dy1 = np.diff(Y,axis=1)
    dy2 = np.diff(Y,axis=2)
    dz0 = np.diff(Z,axis=0)
    dz1 = np.diff(Z,axis=1)
    dz2 = np.diff(Z,axis=2)
    
    
    XX = X.copy()
    XX[:-1]-=0.5*dx0
    XX[-1]-=0.5*dx0[-1]
    XX[:,:-1]-=0.5*dx1
    XX[:,-1]-=0.5*dx1[:,-1]
    XX[:,:,:-1]-=0.5*dx2
    XX[:,:,-1]-=0.5*dx2[:,:,-1]
    
    YY = Y.copy()
    YY[:-1]-=0.5*dy0
    YY[-1]-=0.5*dy0[-1]
    YY[:,:-1]-=0.5*dy1
    YY[:,-1]-=0.5*dy1[:,-1]
    YY[:,:,:-1]-=0.5*dy2
    YY[:,:,-1]-=0.5*dy2[:,:,-1]
    
    ZZ = Z.copy()
    ZZ[:-1]-=0.5*dz0
    ZZ[-1]-=0.5*dz0[-1]
    ZZ[:,:-1]-=0.5*dz1
    ZZ[:,-1]-=0.5*dz1[:,-1]
    ZZ[:,:,:-1]-=0.5*dz2
    ZZ[:,:,-1]-=0.5*dz2[:,:,-1]
    
    XT[:-1,:-1,:-1]=XX.copy()
    YT[:-1,:-1,:-1]=YY.copy()
    ZT[:-1,:-1,:-1]=ZZ.copy()
    
    
    XT[-1,:-1,:-1]=XT[-2,:-1,:-1]+dx0[-1]
    XT[:-1,-1,:-1]=XT[:-1,-2,:-1]+dx1[:,-1,:]
    XT[:-1,:-1,-1]=XT[:-1,:-1,-2]+dx2[:,:,-1]
    XT[:-1,-1,-1]=0.5*(XT[:-1,-1,-2]+dx2[:,-1,-1]+XT[:-1,-2,-1]+dx1[:,-1,-1])
    XT[-1,:-1,-1]=0.5*(XT[-1,:-1,-2]+dx2[-1,:,-1]+XT[-2,:-1,-1]+dx0[-1,:,-1])
    XT[-1,-1,:-1]=0.5*(XT[-1,-2,:-1]+dx1[-1,-1,:]+XT[-2,-1,:-1]+dx0[-1,-1,:])
    XT[-1,-1,-1]=(XT[-1,-2,-1]+dx1[-1,-1,-1]+XT[-2,-1,-1]+dx0[-1,-1,-1]+XT[-1,-1,-2]+dx2[-1,-1,-1])/3
    
    YT[-1,:-1,:-1]=YT[-2,:-1,:-1]+dy0[-1]
    YT[:-1,-1,:-1]=YT[:-1,-2,:-1]+dy1[:,-1,:]
    YT[:-1,:-1,-1]=YT[:-1,:-1,-2]+dy2[:,:,-1]
    YT[:-1,-1,-1]=0.5*(YT[:-1,-1,-2]+dy2[:,-1,-1]+YT[:-1,-2,-1]+dy1[:,-1,-1])
    YT[-1,:-1,-1]=0.5*(YT[-1,:-1,-2]+dy2[-1,:,-1]+YT[-2,:-1,-1]+dy0[-1,:,-1])
    YT[-1,-1,:-1]=0.5*(YT[-1,-2,:-1]+dy1[-1,-1,:]+YT[-2,-1,:-1]+dy0[-1,-1,:])
    YT[-1,-1,-1]=(YT[-1,-2,-1]+dy1[-1,-1,-1]+YT[-2,-1,-1]+dy0[-1,-1,-1]+YT[-1,-1,-2]+dy2[-1,-1,-1])/3
    
    ZT[-1,:-1,:-1]=ZT[-2,:-1,:-1]+dz0[-1]
    ZT[:-1,-1,:-1]=ZT[:-1,-2,:-1]+dz1[:,-1,:]
    ZT[:-1,:-1,-1]=ZT[:-1,:-1,-2]+dz2[:,:,-1]
    ZT[:-1,-1,-1]=0.5*(ZT[:-1,-1,-2]+dz2[:,-1,-1]+ZT[:-1,-2,-1]+dz1[:,-1,-1])
    ZT[-1,:-1,-1]=0.5*(ZT[-1,:-1,-2]+dz2[-1,:,-1]+ZT[-2,:-1,-1]+dz0[-1,:,-1])
    ZT[-1,-1,:-1]=0.5*(ZT[-1,-2,:-1]+dz1[-1,-1,:]+ZT[-2,-1,:-1]+dz0[-1,-1,:])
    ZT[-1,-1,-1]=(ZT[-1,-2,-1]+dz1[-1,-1,-1]+ZT[-2,-1,-1]+dz0[-1,-1,-1]+ZT[-1,-1,-2]+dz2[-1,-1,-1])/3
    
    
    return XT,YT,ZT

def rotMatrix(v,theta,deg=True):
    """ Generalized rotation matrix.
    
    Args:
        
        - v (list): Rotation axis around which matrix rotates
        
        - theta (float): Rotation angle (by default in degrees)
        
    Kwargs:
        
        - deg (bool): Whether or not angle is in degrees or radians (Default True)
        
    Returns:
        
        - 3x3 matrix rotating points around vector v by amount theta.
    """
    if deg==True:
        theta = np.deg2rad(theta.copy())
    v/=np.linalg.norm(v)
    m11 = np.cos(theta)+v[0]**2*(1-np.cos(theta))
    m12 = v[0]*v[1]*(1-np.cos(theta))-v[2]*np.sin(theta)
    m13 = v[0]*v[2]*(1-np.cos(theta))+v[1]*np.sin(theta)
    m21 = v[0]*v[1]*(1-np.cos(theta))+v[2]*np.sin(theta)
    m22 = np.cos(theta)+v[1]**2*(1-np.cos(theta))
    m23 = v[1]*v[2]*(1-np.cos(theta))-v[0]*np.sin(theta)
    m31 = v[0]*v[2]*(1-np.cos(theta))-v[1]*np.sin(theta)
    m32 = v[1]*v[2]*(1-np.cos(theta))+v[0]*np.sin(theta)
    m33 = np.cos(theta)+v[2]**2*(1-np.cos(theta))
    return np.array([[m11,m12,m13],[m21,m22,m23],[m31,m32,m33]])


def Norm2D(v):
    reciprocal = np.abs(1/v)
    if np.isclose(reciprocal[0],reciprocal[1]):
        return v*reciprocal[0]
    
    ratio = np.max(reciprocal)/np.min(reciprocal)
    if np.isclose(np.mod(ratio,1),0.0) or np.isclose(np.mod(ratio,1),1.0):
        return v*np.min(reciprocal)*ratio
    else:
        return v

def LengthOrder(v):
    nonZeroPos = np.logical_not(np.isclose(v,0.0))
    if np.sum(nonZeroPos)==1:
        Rv = v/np.linalg.norm(v)
        return Rv
    if np.sum(nonZeroPos)==0:
        raise AttributeError('Provided vector is zero vector!')
    
    if np.sum(nonZeroPos)==3:
        v1 = Norm2D(v[:2])
        ratio = v1[0]/v[0]
        v2 = Norm2D(np.array([v1[0],v[2]*ratio]))
        ratio2 = v2[0]/v1[0]
        Rv = np.array([v2[0],v1[1]*ratio2,v2[1]])
    else:
        Rv = np.zeros(3)
        nonZeros = v[nonZeroPos]
        Rv[nonZeroPos] = Norm2D(nonZeros)
    
    if not np.isclose(np.dot(Rv,v)/(np.linalg.norm(Rv)*np.linalg.norm(v)),1.0):
        raise AttributeError('The found vector is not parallel to original vector: {}, {}',format(Rv,v))
    return Rv

def overWritingFunctionDecorator(overWritingFunction):
    def overWriter(func):
        return overWritingFunction
    return overWriter


@KwargChecker()
def vectorAngle(V1,V2):
    """calculate angle in radians between V1 and V2.
    
    Args:
    
        - V1 (list): List or array of numbers
        
        - V2 (list): List or array of numbers
        
    Return:
        
        - theta (float): Angle in degrees between the two vectors
    """
    return np.arccos(np.dot(V1,V2.T)/(np.linalg.norm(V1)*np.linalg.norm(V2)))

def normlength(V):
    """rescale V to have unit length"""
    return V/np.linalg.norm(V)


def invert(M):
    """Invert non-square matrices as described on https://en.wikipedia.org/wiki/Generalized_inverse.
    
    Args:
        
        - M (matrix): Matrix in question.
        
    Returns:
        
        - Left or right inverse matrix depending on shape of provided matrix.
    """
    s = M.shape
    if s[0]>s[1]:
        return np.dot(np.linalg.inv(np.dot(M.T,M)),M.T)
    else:
        return np.dot(M.T,np.linalg.inv(np.dot(M,M.T)))



def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner



class CentreOfMass(object):
    """Small helper class holding a center of gravity and its weight"""
    def __init__(self,position,weight):
        x,y,z = position
        self.x = x
        self.y = y
        self.z = z
        self.weight = weight
        self.originals = [[position,weight]]
        
    def addPoint(self,position,weight=1.0):
        totalWeight = self.weight+weight
        newPosition = np.sum([self.position*self.weight,position*weight],axis=0)/totalWeight
        self.position = newPosition
        self.weight=totalWeight
        self.originals.append([position,weight])
        
    @property
    def position(self):
        return np.array([self.x,self.y,self.z])
    
    @position.setter
    def position(self,newPosition):
        x,y,z = newPosition
        self.x = x
        self.y = y
        self.z = z
        
    def __str__(self):
        return "CentreOfMass at ({},{},{}) with weight {}".format(*self.position,self.weight)

    def __repr__(self) -> str:
        return 'CentreOfMass(position=[{},{},{}],weight={})'.format(*self.position,self.weight)
        
def distance(a,b,dx=1,dy=1,dz=1):
    """Calculate distance with variable metric"""
    return np.linalg.norm(np.dot(a-b,[dx,dy,dz]))



def clusterPoints(positions,weights=None,distanceThreshold=0.01, shufflePoints=True, distanceFunction=None, fileName = None):
    """Combine positions within distance threshold into centres of gravity with the provided weights
    
    Args:
        
        - positions (list [n,3]): List of positions to be combined
        
    Kwargs:
        
        - weights (list [n]): Weights corresponding to positions (default None -> ones)
        
        - distanceThreshold (float): Distance within which points are to be combined (default 0.01)
        
        - shufflePoints (bool): If True, shuffle the provided positions and correspondingly their weights (default False)
        
        - distanceFunction (function): Function to calculate distance (default None -> np.linalg.norm)
        
    Returns:
        
        - 
        
    Note:
        
        In situations where the positions provided are correlated it can happen that 
        the found centres of gravity are different than expected.
        
    
    """
    positions = np.asarray(positions)
    
    if len(positions) == 0:
        if fileName is not None:
            print('no peak positions found in {}, too high threshold'.format(str(fileName)))
        else:
            print('no peak positions found, too high threshold')

    if distanceFunction is None:
        distanceFunction = lambda a,b: np.linalg.norm(a-b)
    if weights is None:
        weights = np.zeros(len(positions))
    else:
        weights = np.asarray(weights)
        
    if shufflePoints:
        shuffled = np.concatenate([positions,weights.reshape(-1,1)],axis=1)
        np.random.shuffle(shuffled)
        positions = shuffled[:,:3]
        weights = shuffled[:,-1]
       
    if len(positions) == 0:
        return []
    centres = [CentreOfMass(weight=weights[0],position=positions[0])]
    
    for I,(pos,weight) in enumerate(zip(positions[1:],weights[1:])):
        #if np.mod(I,100):
            #print(I,'len(peak) = ',len(peaks))
        posUsed = False
        for p in centres:
            if distanceFunction(p.position,pos)<distanceThreshold:
                p.addPoint(pos,weight=weight)
                posUsed = True
                break
            
        if not posUsed:
            centres.append(CentreOfMass(pos,weight))
    return centres


def calculateTriplets(reflections,normalized=False):
    """Calculate cross product triples between all points
    
    Args: 
        
        - reflections (list [n,3]): List of reflections to use
        
    Kwargs:
        
        - normalized (bool): Normalize the length of normal (default False)
        
    """
    tripletNormal = []
     
    points = [np.squeeze(a) for a in np.vsplit(reflections, reflections.shape[0])]
    np.random.shuffle(points)
    # Calculate the cross product of the vectors connecting 3 random points
    
    for a1, a2, a3 in product(points, repeat=3):
        # find cross product
        normal = np.cross(a2 - a1, a3 - a1)
        
        # if length of cross product is 0, continue
        if np.allclose(normal, 0, atol=1e-2):
            continue
        if normalized:
            # make it normalized and make it 'mostly positive'
            normal /= np.linalg.norm(normal)
            normal*=np.sign(np.sum(normal))
        
        tripletNormal.append(normal)
    return tripletNormal


def plusMinusGenerator():
    """generator giving an infinite series following 0, -1, 1, -2, 2, ..."""

    yield 0
    start = 0
    while True:
        if start>0:
            start=-start
        else:
            start=-start+1
        yield start


def calculateHKLWithinQLimitsGenerator(BMatrix,QMin=0,QMax=10):
    """Generator to calculate all HKLs within a range of Qs"""
    for h in plusMinusGenerator():
        for k in plusMinusGenerator():
            for l in plusMinusGenerator():
                q = np.linalg.norm(np.dot(BMatrix,[h,k,l]))
                if (q>QMax*1.5):
                    break
                if q>QMin and q<QMax:
                    yield [h,k,l]
            if np.linalg.norm(np.dot(BMatrix,[h,k,0]))>QMax*1.5:
                break
        if np.linalg.norm(np.dot(BMatrix,[h,0,0]))>QMax*1.5:
            break
    



def calculateHKLWithinQLimits(BMatrix,QMin=0,QMax=10):
    positions = []
    
    for h in plusMinusGenerator():
        for k in plusMinusGenerator():
            for l in plusMinusGenerator():
                q = np.linalg.norm(np.dot(BMatrix,[h,k,l]))
                if (q>QMax*1.5):
                    break
                if q>QMin and q<QMax:
                    positions.append([h,k,l])
            if np.linalg.norm(np.dot(BMatrix,[h,k,0]))>QMax*1.5:
                break
        if np.linalg.norm(np.dot(BMatrix,[h,0,0]))>QMax*1.5:
            break
    return positions


def saveSampleToDesk(sample,fileName):
    with open(fileName, 'wb') as handle:
        pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def loadSampleFromDesk(fileName):
    with open(fileName, 'rb') as handle:
        sample = pickle.load(handle)
    
    return sample


def giveUnitCellToHDF(filePath,unitCell):
    """
    adds unit cell to hdf files

    filePath (list): files that unit cell will be added to

    unitCell (list): unit cell parameters

    """

    for file in filePath:
        with hdf.File(file,mode='r+') as f:
            sample = f.get('/entry/sample')
            try:
                sample.create_dataset('unit_cell',data = unitCell)
            except ValueError:
                print('Unit cell already added to: ',file)

def without_keys(dictionary, keys): # Remove key word argument from kwargs
    return {x: dictionary[x] for x in dictionary if x not in keys}


def arange(start,stop,step):
        stepsTaken = 0
        while start+step*(stepsTaken+1)<stop:
            yield (start+step*stepsTaken,start+step*(stepsTaken+1))
            stepsTaken+=1
            
        yield(start+step*stepsTaken,stop)


def calculateRotationMatrixAndOffset(points):
    
    v1, v2, v3 = points
    dV1 = v2-v1
    dV2 = v3-v1
    
    dV1*=1.0/np.linalg.norm(dV1)
    dV2*=1.0/np.linalg.norm(dV2)
    
    
    N = np.cross(dV2,dV1)
    N *= 1.0/np.linalg.norm(N)
    
    rotVector = np.cross([0.0,0.0,1.0],N)
    if np.isclose(np.linalg.norm(rotVector),0): # if they are parallel no rotation is needed
        theta = 0.0
        rotVector = np.array([0.0,0.0,1.0])
    else:
        #rotVector *=1.0/np.linalg.norm(rotVector)
        theta = np.arccos(np.dot([0.0,0.0,1.0],N))#/np.linalg.norm(rotVector)) #
    
    
    Rot3D = rotMatrix(rotVector, -theta,deg=False)
    
    v1m,v2m,v3m = np.einsum('ij,...j->...i',Rot3D,points)
    
    dV1m = v2m-v1m
    dV2m = v3m-v1m
    Nm = np.cross(dV2m,dV1m)
    Nm *= 1.0/np.linalg.norm(Nm)
    
    offsetm = [np.dot(Nm,x) for x in [v1m,v2m,v3m]]
    
    if not np.all(np.isclose(offsetm,offsetm[0])):
        raise AttributeError('Calculated plane does not have the defining points in the same distance from...')
    
    ## in plane rotation, so that v1m is || to x
    thetaInPlan = np.arctan2(*-dV1m[-2::-1]) # take first two entries and flip the, i.e. y,x
    Rot3DInPlane = rotMatrix(Nm, -thetaInPlan,deg=False)
    
    totalRotMat = np.dot(Rot3DInPlane,Rot3D)
    return totalRotMat,-offsetm[0]


def calculateRotationMatrixAndOffset2(points):
    v1, v2, v3 = points
    Q1 = v2-v1
    Q2 = v3-v1

    # projectionVectors = np.array([[0,0,1],[1,1,0],[1,-1,0]])
    axisVectors = np.eye(3)
    ## Assume that Q1/HKL1 is along x-axis

    Alpha1 = np.rad2deg(np.arccos(np.dot(Q1,axisVectors[0])/(np.linalg.norm(Q1))))

    if np.isclose(Alpha1,0.0) or np.isclose(Alpha1,180.0): # Q1 is parallel to 1 0 0
        Rot1 = np.array([0.0,0.0,1.0])
    else:
        Rot1 = np.cross(Q1,axisVectors[0])

        Rot1*=1.0/np.linalg.norm(Rot1)
    ROT1 = rotMatrix(Rot1,Alpha1)

    # Rotate Q2 into Q1's frame
    Q2Rot = np.dot(ROT1,Q2)
    Q2Rot-= np.dot(axisVectors[0],Q2Rot)*axisVectors[0]# project out [1,0,0] as this rotation has been done by Q1

    Alpha2 = np.rad2deg(np.arccos(np.dot(Q2Rot,axisVectors[1])/(np.linalg.norm(Q2Rot))))#np.rad2deg(np.arccos(Q2Rot[1]/np.linalg.norm(Q2Rot)))

    if np.isclose(Alpha2,0.0) or np.isclose(Alpha2,180.0):
        Rot2 = np.array([0.0,0.0,1.0])
    else:
        Rot2 = np.cross(Q2Rot,axisVectors[1])
        Rot2*=1.0/np.linalg.norm(Rot2)
    ROT2 = rotMatrix(Rot2,Alpha2)
    ROT = np.dot(ROT2,ROT1)
    offset = np.einsum('ij,...j->...i',ROT,[v1, v2, v3])[:,2]
    if not np.all(np.isclose(offset,offset[0])):
        raise AttributeError('Calculated plane does not have the defining points in the same distance from...')
    
    return ROT,offset.mean()


def merge(dataFilesList,saveFileName,directory=None, A3Tolerance=0.05, A4Tolerance = 0.1, wavelengthTolerance = 0.01):
    """Merge multiple single crystal data files togehter with equal sample, A4, and wavelength
    
    Args:
        
        - dataFileList(list): List of data file paths to be merged
        
        - saveFileName(string): Name or path to output file
        
    Kwargs:
        
        - directory (path): Saving directory. If non use saveFileName as full path (default None)
        
        - A3Tolerance (float): Tolerance of A3 in degrees where points are merged (default 0.05)
        
        - A4Tolerance (float): Tolerance of A4. If files are not within tolerance merging will fail (default 0.1)
        
        - wavelengthTolerance (float): Tolerance of wavelength. If files are not within tolerance merging will fail (default 0.01)
        
    Raises:
        
        - AttributeError
        
    Copies first provided data file into the new path and changes counts, A3, monitor, time, and summed counts as needed. That is
    if a large tolerance in either A4 or wavelength is provided, the values for the first data file is used.
    
    """
    #### INPUT
    if directory is None:
        directory, savefileName = os.path.split(saveFileName)
        
    
    # Perform checks
    equalParameters = ['twoThetaPosition','wavelength']
    equalParametersTolerance = [A4Tolerance,wavelengthTolerance]
    files = shallowRead(dataFilesList,equalParameters)
    
    trueValue = None
    truthTable = []
    for name,tol in zip(equalParameters,equalParametersTolerance):
        trueValue = None
        localTruthTable = []
        for file in files:
            if trueValue is None:
                trueValue = file[name]
            else:
                test = np.isclose(trueValue,file[name],atol=tol)
                if hasattr(test,'__len__'):
                    test= np.all(test)
                localTruthTable.append(test)
            
            
        truthTable.append(localTruthTable)
    
    differentFiles = np.asarray(truthTable)
    parameterwiceTruth = np.all(truthTable,axis=1) # If any is false, we have a problem Houston
    
    if np.any(np.logical_not(parameterwiceTruth)):
        errorMessage = 'Merging of data files not possible as following parameter(s) are outside tolerance:'
        parameterText = '\n'.join([name+'(tol: {}): '.format(tol)+str(truthtab) for name,tol,truthtab in zip(equalParameters,equalParametersTolerance,truthTable)])
        differentFiles = 'File(s) being different from first: '+', '.join(np.asarray(dataFilesList[1:])[np.any(np.logical_not(truthTable),axis=0)])
        raise AttributeError('\n'.join([errorMessage,parameterText,differentFiles]))
    
    # Copy first data file into new file as a base
    savepath = os.path.join(directory,savefileName)
    if not os.path.exists(directory): # If save folder does not exist, please make it
        os.mkdir(directory)
    
    #originalPosition
    shutil.copy(dataFilesList[0],savepath)
    
    # Find min and max of A3 as well as average across all files
    A3files = shallowRead(dataFilesList,['A3'])
    
    # Check if length is 1, then it is a powder!!!
    powderFiles = [f['file'] for f in A3files if len(f['A3'])<2 ]
    if len(powderFiles)>0:
        raise AttributeError('Following file(s) consist(s) of only one A3 step:\n'+'\n'.join(powderFiles))
        
    A3Start = []
    A3Stop = []
    A3Step = []
    
    meanDiff = lambda x: np.mean(np.diff(x)).flatten()[0]
    
    totalA3 = []
    
    for a3f in A3files:
        a3 = a3f['A3']
        totalA3.append(a3)
        start,stop,step = [f(a3) for f in [np.min,np.max,meanDiff]]
        A3Start.append(start)
        A3Stop.append(stop)
        A3Step.append(step)
        #print(start,stop,step)
    
    A3Start = np.asarray(A3Start)
    A3Stop = np.asarray(A3Stop)
    A3Step = np.asarray(A3Step)
    
    totalA3 = np.asarray(np.concatenate(totalA3))
    newA3 = np.unique(np.round(totalA3/A3Tolerance))*A3Tolerance
    
    
    
    def getPositionInFile(file,parameter):
        """ Get position of specified parameter in HDF file"""
        if not parameter in HDFTranslationAlternatives:
                intensityPositionsInFile = [HDFTranslation[parameter]]
        else:
            intensityPositionsInFile = HDFTranslationAlternatives[parameter]
            
        if len(intensityPositionsInFile)>1: # If there are multiple possible positions
            for pos in intensityPositionsInFile[::-1]: # Last entry is the newest
                if not saveFile.get(pos) is None:
                    intensityNotFound = False
                    break
            else:
                intensityNotFound = True
        else:
            if saveFile.get(intensityPositionsInFile[0]) is None:
                intensityNotFound = True
                pos = intensityPositionsInFile
                intensityPositionsInFile = list(intensityPositionsInFile)
                
            else:
                intensityNotFound = False
                pos = intensityPositionsInFile[0]
                
        if intensityNotFound:
    
            raise AttributeError('Could not find any intensity matrix in first data file. Looked at:\n'+'\n'.join(intensityPositionsInFile))
            
        return '/'+pos
    
    totalSteps = len(newA3)
    with hdf.File(savepath,'r') as saveFile:
        # Find positions in dataFilesList
        countPositionInFile = '/'+HDFCounts#
        countShape = saveFile[countPositionInFile].shape
        monitorPositionInFile = getPositionInFile(saveFile,'monitor')
        summedCountsPositionInFile = getPositionInFile(saveFile,'summedCounts')
        timePositionInFile = getPositionInFile(saveFile,'time')
    
    
    time = np.zeros((totalSteps),dtype=float)
    summedCounts =  np.zeros((totalSteps),dtype=float)
    monitor = np.zeros((totalSteps),dtype=float)
    counts = np.zeros((totalSteps,*countShape[1:]),dtype=float)
    
    
    compressionValues = []
    for file in A3files:
        indices = np.asarray([np.argmin(np.abs(newA3-localA3)) for localA3 in file['A3']])
        with hdf.File(file['file'],'r') as f:
            
            fileMonitor = np.asarray(f.get(monitorPositionInFile))
            fileTime = np.asarray(f.get(timePositionInFile))
            fCounts = f.get(countPositionInFile)
            compressionValues.append(fCounts.compression_opts)
            fileCounts = np.asarray(fCounts)
            
        
        np.add.at(monitor, indices, fileMonitor)#fileMonitor)
        np.add.at(time, indices, fileTime)#fileMonitor)
        np.add.at(counts, indices, fileCounts)#fileMonitor)
        np.add.at(summedCounts, indices, np.sum(fileCounts,axis=(1,2)))#fileMonitor)
        print('Data from {:} loaded'.format(file['file']))
    
    print('All data loaded, saving to {:}'.format(savepath))
    
    with hdf.File(savepath,'r+') as saveFile:
        
        # remove monitor first and replace with correct size
        del saveFile[monitorPositionInFile]
        del saveFile[countPositionInFile]
        del saveFile[timePositionInFile]
        del saveFile[summedCountsPositionInFile]
        
        # And of course save a3 as well... who would forget to do so?
        A3Position = getPositionInFile(saveFile,'A3')
        del saveFile[A3Position]
        
        saveFile.create_dataset(name = A3Position, data=newA3, dtype=float)
        
        saveFile.create_dataset(name = monitorPositionInFile, data=monitor, dtype=float)
        saveFile.create_dataset(name = countPositionInFile, shape=counts.shape, data=counts, dtype=float,compression=6)
        saveFile.create_dataset(name = timePositionInFile, data=time, dtype=float)
        saveFile.create_dataset(name = summedCountsPositionInFile, data=summedCounts, dtype=float)
        
        process = saveFile.create_group('entry/reduction')
        process.attrs['NX_class']=b'NXprocess'
        proc = process.create_group('DMCpy_algorithm_merge')
        proc.attrs['NX_class']=b'NXprocess'
        author= proc.create_dataset('author',shape=(1,),dtype='S70',data=np.string_('DMCpy'))
        author.attrs['NX_class']=b'NX_CHAR'
        author= proc.create_dataset('version',shape=(1,),dtype='S70',data=np.string_(DMCpy.__version__))
        author.attrs['NX_class']=b'NX_CHAR'
        
        date= proc.create_dataset('date',shape=(1,),dtype='S70',data=np.string_(datetime.datetime.now()))
        date.attrs['NX_class']=b'NX_CHAR'
        
        description = proc.create_dataset('description',shape=(1,),dtype='S70',data=np.string_('Merging of equivalent data files where only A3 differs.'))
        description.attrs['NX_class']=b'NX_CHAR'
        
        
        rawData = [os.path.split(df)[-1] for df in dataFilesList]
        proc.create_dataset('rawdata',shape=(len(dataFilesList),),data=np.asarray(rawData,dtype='S'))
        
        print('Data merged and saved in {:}'.format(savepath))


def histogramdd(sample, bins, weights, returnCounts = False):
    """
    Restricted version of numpys multidimensional histogram function. 

    Args:

        - sample (n x m array): Position in m-dimensional space

        - bins (m list): List of bins

        - weights (list): List of weights where each entry has the length n

    Kwargs:

        - returnCounts (bool): if True return also number of entries in each bin (default False)

    """

    try:
        # Sample is an ND-array.
        N, D = sample.shape
    except (AttributeError, ValueError):
        # Sample is a sequence of 1D arrays.
        sample = np.atleast_2d(sample).T
        N, D = sample.shape


    try:
        M = len(bins)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except TypeError:
        # bins is an integer
        bins = D*[bins]


    nbin = np.empty(D, int)
    edges = D*[None]
    for i in range(D):
        edges[i] = np.asarray(bins[i])
        nbin[i] = len(edges[i])+1
    

    # Compute the bin number each sample falls into.
    Ncount = tuple(
        # avoid np.digitize to work around gh-11022
        np.searchsorted(edges[i], sample[:, i], side='right')
        for i in range(D)
    )

    # Using digitize, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right edge to be
    # counted in the last bin, and not as an outlier.
    for i in range(D):
        # Find which points are on the rightmost edge.
        on_edge = (sample[:, i] == edges[i][-1])
        # Shift these points one bin to the left.
        Ncount[i][on_edge] -= 1

    # Compute the sample indices in the flattened histogram matrix.
    # This raises an error if the array is too large.
    xy = np.ravel_multi_index(Ncount, nbin)

    # Compute the number of repetitions in xy and assign it to the
    # flattened histmat.

    histograms = []
    for w in weights:
        hist = np.bincount(xy, w, minlength=nbin.prod())

        # Shape into a proper matrix
        hist = hist.reshape(nbin)

        # This preserves the (bad) behavior observed in gh-7845, for now.
        hist = hist.astype(w.dtype)#, casting='safe')

        # Remove outliers (indices 0 and -1 for each dimension).
        core = D*(slice(1, -1),)
        hist = hist[core]
        histograms.append(hist)

    if returnCounts:
        hist = np.bincount(xy, minlength=nbin.prod())

        # Shape into a proper matrix
        hist = hist.reshape(nbin)

        # This preserves the (bad) behavior observed in gh-7845, for now.
        hist = hist.astype(int)#, casting='safe')

        # Remove outliers (indices 0 and -1 for each dimension).
        core = D*(slice(1, -1),)
        hist = hist[core]
        histograms.append(hist)

    return histograms

def findOrthogonalBasis(v1,v2,v3,B):
    """Calculate an orthogonal basis from projection vectors and B matrix"""
    p1 = LengthOrder(v1)
    p2 = v2
    p1Q = np.dot(B,p1)
    p2Q = np.dot(B,p2)
    
    p2 = LengthOrder(np.dot(np.linalg.inv(B),np.cross(np.cross(p1Q,p2Q),p1Q)))
    p2Q = np.dot(B,p2)
    p3 = LengthOrder(np.dot(np.linalg.inv(B),np.cross(p1Q,p2Q)))      
    return np.asarray([p1,p2,p3])