# SPDX-License-Identifier: MPL-2.0
import copy
import functools
import os
import sys
from operator import sub

pythonVersion = sys.version_info[0]

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import matplotlib.pyplot as plt
#if pythonVersion == 3: # Only for python 3
import matplotlib.ticker as mticker

import numpy as np
from DMCpy import _tools
from mpl_toolkits.axisartist import SubplotHost
try:
    from mpl_toolkits.axisartist.grid_helper_curvelinear import \
        GridHelperCurveLinear
except ImportError:
    from mpl_toolkits.axisartist.floating_axes.grid_helper_curvelinear import \
        GridHelperCurveLinear



def calculateBase(l,span,ticks):
    """Calcualte the tick mark base suitable for current span and ticks

    Args:

        - l (grid locator): Matplotlib grid locator

        - span (float): Width of view

        - ticks (int): Number of ticks wanted

    Returns:

        - base (float): Closest base number according to l.multiples
    """
    ytickorder = np.ceil(np.log10(span/ticks))
    minimalMultiplesy = np.argmin(np.abs(np.power(10,-ytickorder)*span/ticks-l.multiples))
    return l.multiples[minimalMultiplesy]*np.power(10,ytickorder)
    


def updateAxisDecorator(ax,direction='both'):
    def axisDecorator(func):
        @functools.wraps(func)
        def newFunc(*args,**kwargs):
            returnval = func(*args,**kwargs)
            ax.axisChanged(direction=direction)
            return returnval
        return newFunc
    return axisDecorator

def updateXAxisDecorator(ax):
    def axisDecorator(func):
        @functools.wraps(func)
        def newFunc(*args,**kwargs):
            returnval = func(*args,**kwargs)
            ax.xAxisChanged()
            return returnval
        return newFunc
    return axisDecorator

class MaxNLocator(mticker.MaxNLocator):
    def __init__(self, nbins=10, steps=None,
                trim=True,
                integer=False,
                symmetric=False,
                prune=None):
        # trim argument has no effect. It has been left for API compatibility
        super(MaxNLocator,self).__init__(nbins, steps=steps,
                                    integer=integer,
                                    symmetric=symmetric, prune=prune)
        self.create_dummy_axis()
        self._factor = 1#0.0#None

    def __call__(self, v1, v2): # pragma: no cover
        if self._factor is not None:
            self.set_bounds(v1*self._factor, v2*self._factor)
            locs = mticker.MaxNLocator.__call__(self)
            return np.array(locs), len(locs), self._factor
        else:
            self.set_bounds(v1, v2)
            locs = mticker.MaxNLocator.__call__(self)
            return np.array(locs), len(locs)

    def set_factor(self, f):
        self._factor = f


class MultipleLocator(mticker.MultipleLocator):
    def __init__(self,base=None):
        if base is None:
            base = 0.25
        super(MultipleLocator,self).__init__(base)
        self.create_dummy_axis()
        self._factor = 1#0.0
        self._multiplerVals = np.array([1,2,4,5,10])
        self.multiples = 1.0/self.multiplerVals

    @property
    def multiplerVals(self):
        return self._multiplerVals

    @multiplerVals.getter
    def multiplerVals(self):
        return self._multiplerVals

    @multiplerVals.setter
    def multiplerVals(self,multiplerVals):
        self._multiplerVals = multiplerVals
        self.multiples = 1.0/multiplerVals


    def __call__(self, v1, v2): # pragma: no cover
        if self._factor is not None:
            self.axis.set_view_interval(v1*self._factor, v2*self._factor)
            locs = mticker.MultipleLocator.__call__(self)
            

            return np.array(locs), len(locs), self._factor
        else:
            self.axis.set_view_interval(v1, v2)
            locs = mticker.MultipleLocator.__call__(self)
            return np.array(locs), len(locs), None

    def set_factor(self, f):
        self._factor = f

def forceGridUpdate(self):
    self._grid_helper._force_update = True
    self.pchanged()
    self.stale = True

def get_aspect(ax):
    figW, figH = ax.get_figure().get_size_inches()
    _, _, w, h = ax.get_position().bounds
    disp_ratio = (figH * h) / (figW * w)
    data_ratio = sub(*ax.get_ylim()) / sub(*ax.get_xlim())
    return (disp_ratio / data_ratio)

def axisChanged(event,axis=None,forceUpdate=False,direction='both'):
    """Function to recalculate the base number for RLU axis"""
    if axis is None:
        axis = event
    s = axis.sample
    xlim = axis.get_xlim()
    ylim = axis.get_ylim()
    xlimDiff = np.diff(xlim)
    ylimDiff = np.diff(ylim)
    
    if direction == 'both':
        direction = ['x','y']
        difference = [xlimDiff,ylimDiff]
        locators = [axis._grid_helper.grid_finder.grid_locator1,axis._grid_helper.grid_finder.grid_locator2]
    else:
        if direction == 'x':
            difference = [xlimDiff]  
            locators = [axis._grid_helper.grid_finder.grid_locator1]
        else:
            difference = [ylimDiff]  
            locators = [axis._grid_helper.grid_finder.grid_locator2]
        direction = [direction]
    for d,diff,locator in zip(direction,difference,locators):
        forceUpdate = True
        if isinstance(locator,MultipleLocator):
            if not np.isclose(getattr(axis,'_old{}limDiff'.format(d.capitalize())),diff) \
            or forceUpdate:# if new image size

                points = np.array(np.meshgrid(xlim,ylim)).reshape(-1,2) # Generate the 4 corner points
                Qs = np.array([s.inv_tr(p[0],p[1]) for p in points])
                if d=='x':
                    span = np.max(Qs[:,0])-np.min(Qs[:,0])
                else:
                    span = np.max(Qs[:,1])-np.min(Qs[:,1])
                if not hasattr(axis,'{}ticks'.format(d)):
                    setattr(axis,'{}ticks'.format(d),7)
                
                ticks = getattr(axis,'{}ticks'.format(d))
                setattr(axis,'_old{}limDiff'.format(d.capitalize()),diff)
            

            if not hasattr(axis,'{}base'.format(d)):
                base = calculateBase(locator,span,ticks)
            else:
                base = getattr(axis,'{}base'.format(d))

            locator.set_params(base=base)
        
        elif isinstance(locator,MaxNLocator):
            if hasattr(axis,'{}ticks'.format(d)):
                ticks = getattr(axis,'{}ticks'.format(d))
            else:
                ticks = 7
            locator.set_params(nbins = ticks)
        else:
            return 
    
    # force an update
    axis.forceGridUpdate()



def createRLUAxes(self,sample=None, figure=None,ids=[1, 1, 1],basex=None,basey=None,step=0):
    """Create a reciprocal lattice plot for a given DataSet object.
    
    Args:
        
        - Dataset (DataSet): DataSet object for which the RLU plot is to be made.

    Kwargs:

        - figure: Matplotlib figure in which the axis is to be put (default None)

        - ids (array): List of integer numbers provided to the SubplotHost ids attribute (default [1,1,1])

        - basex (float): Ticks are positioned at multiples of this value along x (default None)

        - basey (float): Ticks are positioned at multiples of this value along y (default None)

        - step (float): Projection along out-of-plane projection vector (default 0)

    Returns:
        
        - ax (Matplotlib axes): Axes containing the RLU plot.

    .. note::
        When rlu axis is created, the orientation of Qx and Qy is assumed to be rotated as well. 
        This is to be done in the self.View3D method call!

    .. note::
        When using python 2 the changing of tick marks is not supported due to limitations in matplotlib. However, if python 3 is used, the number 
        of ticks and their location can be change after the initialization using the set_xticks_number, set_yticks_number chaning the wanted number 
        of tick marks, or the set_xticks_base or set_yticks_base to change the base number, see RLU tutorial under Tools. As default a sufficient base
        number is found and will update when zooming.
        
    """
    if sample is None:
        sample = copy.deepcopy(self[0].sample)
        

    if figure is None:
        fig = plt.figure(figsize=(7, 4))
    else:
        fig = figure
    def calculateTicks(ticks,angle,round=True):
        val = ticks/np.tan(angle/2.0)
        if round:
            return np.array(np.round(val),dtype=int)
        else:
            return val


    if  not basex is None or not basey is None: # Either basex or basey is provided (or both)
        if basex is None:
            basex = calculateTicks(1.0,lambda: sample.projectionAngle(),round=False)
        elif basey is None:
            basey = basex/calculateTicks(1.0,lambda: sample.projectionAngle(),round=False)

        grid_locator1 = MultipleLocator(base=basex)
        grid_locator2 = MultipleLocator(base=basey)
    else:
        basex = 0.5
        basey = 0.5

        grid_locator1 = MultipleLocator(base=basex)
        grid_locator2 = MultipleLocator(base=basey)
        
    grid_helper = GridHelperCurveLinear((lambda x,y: sample.tr(x,y), lambda x,y: sample.inv_tr(x,y))
                                        ,grid_locator1=grid_locator1,grid_locator2=grid_locator2)

    ax = SubplotHost(fig, *ids, grid_helper=grid_helper)
    ax.sample = sample
    ax._step = step

    
    ax.basex = basex
    ax.basey = basey

    def set_axis(ax,v1,v2,*args):
        if args != ():
            points = np.concatenate([[v1,v2],[x for x in args]],axis=0)
        else:
            points = np.array([v1,v2])
            
        if points.shape[1] == 3:
            points = ax.sample.calculateHKLtoProjection(points[:,0],points[:,1],points[:,2]).T
        boundaries = np.array([ax.sample.inv_tr(x[0],x[1]) for x in points])
        ax.set_xlim(boundaries[:,0].min(),boundaries[:,0].max())
        ax.set_ylim(boundaries[:,1].min(),boundaries[:,1].max())
        ax.forceGridUpdate()


    fig.add_subplot(ax)
    ax.set_aspect(1.)
    ax.grid(True, zorder=0)

    if not np.isclose(ax.sample.projectionAngle(),np.pi/2.0,atol=0.001):
        ax.axis["top"].major_ticklabels.set_visible(True)
        ax.axis["right"].major_ticklabels.set_visible(True)

    def advancedFormat_coord(ax,x,y):
        z = np.ones_like(x)*ax._step
        return ax.sample.format_coord(x,y,z)

    ax.format_coord = lambda x,y: advancedFormat_coord(ax,x,y)
    ax.set_axis = lambda v1,v2,*args: set_axis(ax,v1,v2,*args)

    def beautifyLabel(vec):
        Vec = [x.astype(int) if np.isclose(x.astype(float)-x.astype(int),0.0,atol=1e-4) else x.astype(float) for x in vec]
        return '{} [RLU]'.format(', '.join([str(x) for x in Vec]))


    projectionMat = np.delete(ax.sample.projectionVectors,2,axis=1)
    
    ax.set_xlabel(beautifyLabel(projectionMat[:,0]))
    ax.set_ylabel(beautifyLabel(projectionMat[:,1]))



    ax.calculateTicks = lambda value:calculateTicks(value,ax.sample.projectionAngle)
    ax.forceGridUpdate = lambda:forceGridUpdate(ax)
    ax._oldXlimDiff = np.diff(ax.get_xlim())
    ax._oldYlimDiff = np.diff(ax.get_ylim())

    ax.get_aspect_ratio = lambda: get_aspect(ax)

    ax.callbacks.connect('xlim_changed', axisChanged)
    ax.callbacks.connect('ylim_changed', axisChanged)
    ax.get_figure().canvas.mpl_connect('draw_event',lambda event: axisChanged(event,axis=ax,forceUpdate=True))
    ax.axisChanged = lambda direction='both': axisChanged(None,ax,forceUpdate=True,direction=direction)

    @updateAxisDecorator(ax=ax,direction='x')
    def set_xticks_base(xBase,ax=ax):
        """Setter of the base x ticks to be used for plotting

        Args:

            - xBase (float): Base of the tick marks

        """
        if not isinstance(ax._grid_helper.grid_finder.grid_locator1,MultipleLocator):
            l1 = MultipleLocator(base=xBase)
            ax._grid_helper.update_grid_finder(grid_locator1=l1)

        ax.xbase = xBase

    @updateAxisDecorator(ax=ax,direction='y')
    def set_yticks_base(yBase,ax=ax):
        """Setter of the base y ticks to be used for plotting

        Args:

            - yBase (float): Base of the tick marks

        """
        if not isinstance(ax._grid_helper.grid_finder.grid_locator2,MultipleLocator):
            l2 = MultipleLocator(base=yBase)
            ax._grid_helper.update_grid_finder(grid_locator2=l2)
        ax.ybase = yBase

    @updateAxisDecorator(ax=ax,direction='x')
    def set_xticks_number(xNumber,ax=ax):
        """Setter of the number of x ticks to be used for plotting

        Args:

            - xNumber (int): Number of x tick marks

        """
        if not isinstance(ax._grid_helper.grid_finder.grid_locator1,MaxNLocator):
            l1 = MaxNLocator(nbins=xNumber)
            ax._grid_helper.update_grid_finder(grid_locator1=l1)
        ax.xticks = xNumber

    @updateAxisDecorator(ax=ax,direction='y')
    def set_yticks_number(yNumber,ax=ax):
        """Setter of the number of y ticks to be used for plotting

        Args:

            - yNumber (int): Number of y tick marks

        """
        if not isinstance(ax._grid_helper.grid_finder.grid_locator2,MaxNLocator):
            l2 = MaxNLocator(nbins=yNumber)
            ax._grid_helper.update_grid_finder(grid_locator2=l2)
        ax.yticks = yNumber

    ax.set_xticks_base = set_xticks_base
    ax.set_yticks_base = set_yticks_base
    ax.set_xticks_number = set_xticks_number
    ax.set_yticks_number = set_yticks_number


    return ax


