import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np


class InteractiveViewer(object):
    """Interactive viewer for 2D detector and 1D scan variable"""
    def __init__(self,data,twoTheta,pixelPosition,scanValues=None,scanParameter=None,
                 scanValueFormat=None,scanValueUnit=None,colorbar=False,outputFunction=print,
                 mainTitle='Single Step',vmin=None,vmax=None, positive2Theta=True,
                 dataLabel = 'Intensity',axis_1_label='Sum over z',axis_2_label='Sum over 2Theta',
                 xlabel='2Theta [deg]',ylabel='z [cm]',cmap='viridis'):
        """

        args:
            - data (array): 3D array with data data
            
            - twoTheta (array): 2D array holding the 2theta values
            
            - pixelPosition (array): 3D array holding pixel position (x,y,z as function of 2theta and out of plane position)
            
        kwags:
            
            - scanValues (array): 1D array with values of scan parameter (default None)
            
            - scanParameter (str): Name of scanned parameter (default None)
            
            - scanValueFormat (str): Format string for scan parameter values (default None)
            
            - scanValueUnit (str): Unit of scan parameter (default None)
            
            - colorbar (bool): Boolean value for showing color bar (default False)
            
            - outputFunction (function): Function called when clicking on data axis (default print)

            - mainTitle (str): Title of data figure (default 'Single Step'')
                                                          
            - vmin (float): Minimum value for color plot (default None - auto)                                              
            
            - vmax (float): Maximum value for color plot (default None - auto)                                              

            - dataLabel (str): Label of the data axis (default Intensity)
            
            - axis_1_label (str): Label of axis 1 (default Sum over z)
            
            - axis_2_label (str): Label of axis 1 (default Sum over 2theta)
            
            - xlabel (str): Label of x axis (default '2Theta [deg]')
            
            - ylabel (str): Label of y axis (default 'z [cm]')

            - cmap (str): Name of color map (default viridis)

        """
        
        # Initialize index to -1 to ensure plotting of first data
        self.index = -1
        self.data = data.transpose(0,2,1) # Transpose for quicker data plotting
        
        # If scan values are not provided, create [0,1,2,3,...]
        if not scanValues is None:
            self.scanValues = scanValues
        else:
            self.scanValues = np.arange(len(data))
            
        self.dataLabel = dataLabel
        self.axis_1_label = axis_1_label
        self.axis_2_label = axis_2_label
            
        self.cmap = cmap
        self.colorbar = colorbar
        self.mainTitle = mainTitle
        
        self.xlabel = xlabel
        self.ylabel = ylabel
        
        if scanValueFormat is None: # Automatically create slider text format
            if len(self.scanValues) == 1:
                minStep = 1
            else:
                minStep = np.diff(self.scanValues).mean()
            order = int(np.ceil(np.log10(minStep)))
            decimals = np.max([0,3-order])
            self.scanValueFormat  = '{:.'+str(decimals)+'f}'#'{:.'+str(order)+'}'
            self.valfmt = '%.'+str(decimals)+'f'
            if not scanValueUnit is None:
                self.scanValueFormat = self.scanValueFormat + ' ['+scanValueUnit+']'
                self.valfmt = self.valfmt + ' ['+scanValueUnit+']'
            self.scanValueFormat = self.scanValueFormat + ' (step {})'
        # Copy two theta, and create array holding edges between values, including outer most values (length is 1 larger than twoTheta)
        self.twoTheta = twoTheta
        self.twoThetaStep = np.mean(np.diff(twoTheta[0,:]))
        
        if positive2Theta:
            self.twoThetaExtended = np.abs(np.arange(self.twoTheta[0,0]-self.twoThetaStep*0.5,self.twoTheta[0,-1]+self.twoThetaStep*0.6,self.twoThetaStep))
        else:
            self.twoThetaExtended = np.arange(self.twoTheta[0,0]-self.twoThetaStep*0.5,self.twoTheta[0,-1]+self.twoThetaStep*0.6,self.twoThetaStep)
        
        # Repeat for pixel position
        self.pixelPosition = pixelPosition[2]
        self.pixelPositionStep = np.mean(np.diff(self.pixelPosition[:,0]))
        self.pixelPositionExtended = np.arange(self.pixelPosition[0,0]-self.pixelPositionStep*0.5,self.pixelPosition[-1,0]+self.pixelPositionStep*0.6,self.pixelPositionStep)
        
        
        self.scanSteps = len(self.scanValues)
        self.scanValuesStep = minStep
        self.scanValuesExtended = np.arange(self.scanValues[0]-self.scanValuesStep*0.5,self.scanValues[-1]+self.scanValuesStep*0.6,self.scanValuesStep)
        
        self.scanStepExtended = np.arange(-0.5,len(self.data)-1+0.6,1.0)
        
        # Create figure with axes. Layout is 3 x 6
        self.heights = [8,1,6] # make slider thin
        self.widths = [1,3,3,3,3,1] # Change widths
        self.fig = plt.figure(constrained_layout=False,figsize=(13,9))
        
        # Create gridspec
        self.gs = self.fig.add_gridspec(3, 6,width_ratios=self.widths,height_ratios=self.heights)
        self.ax_singleStep = self.fig.add_subplot(self.gs[0, 1:-1])
        
        self.ax_singleStep.set_title(self.mainTitle)
        self.ax_slider = self.fig.add_subplot(self.gs[1, 1:-1])
        self.ax_alphaIntegrated = self.fig.add_subplot(self.gs[2,0:3])
        self.ax_alphaIntegrated.set_title(self.axis_1_label)
        # Share the yaxis between ax_alphaIntegrated and ax_thetaIntegrated
        self.ax_thetaIntegrated = self.fig.add_subplot(self.gs[2,3:],sharey=self.ax_alphaIntegrated)
        self.ax_thetaIntegrated.set_title(self.axis_2_label)
        
        
        if not scanParameter is None:
            scanLabel = scanParameter
        else:
            scanLabel = 'Scan Step'
        
        # Create slider with values between 0 and scanSteps -1)
        if self.scanSteps == 1:
            self.indexSlider = Slider(self.ax_slider, label=scanLabel, valmin=-0.5, valmax=0.5, valinit=0,valfmt=self.valfmt)
        else:
            self.indexSlider = Slider(self.ax_slider, label=scanLabel, valmin=0, valmax=self.scanSteps-1, valinit=-1,valfmt=self.valfmt)
        
        self.indexSlider.on_changed(lambda val: self.sliders_on_changed(val))
        
        
        # add labels
        self.ax_singleStep.set_xlabel(self.xlabel)
        self.ax_singleStep.set_ylabel(self.ylabel)
        
        self.ax_alphaIntegrated.set_ylabel('Scan Step')
        self.ax_alphaIntegrated.set_xlabel(self.xlabel)
        self.ax_thetaIntegrated.set_xlabel(self.ylabel)
        
        # Sum over two theta and alpha(out of plane)
        self.IThetaIntegrated = self.data.sum(axis=1)
        self.IAlphaIntegrated = self.data.sum(axis=2)
        
        if vmin is None:
            vmin = np.nanmin(self.data)
        if vmax is None:
            vmax = np.nanmax(self.data)
            
        self.initialLimits = [vmin,vmax]
        
        # Plotting limits across all steps
        
        # plot theta and alpha integrated intensities

        self.ax_alphaIntegrated._pcolormesh = self.ax_alphaIntegrated.pcolormesh(self.twoThetaExtended,self.scanStepExtended,self.IAlphaIntegrated,cmap=self.cmap)#,vmin=vmin,vmax=vmax)
        self.ax_thetaIntegrated._pcolormesh = self.ax_thetaIntegrated.pcolormesh(self.pixelPositionExtended,self.scanStepExtended,self.IThetaIntegrated,cmap=self.cmap)#,vmin=vmin,vmax=vmax)
        
        #self.ax_alphaIntegrated.axis('auto')
        #self.ax_thetaIntegrated.axis('auto')
        #self.ax_thetaIntegrated.set_yticks([])
        #self.ax_thetaIntegrated.sharey(self.ax_alphaIntegrated)
        
        
        if not scanValues is None: # if scan values are provided
            # Linear interpolation between scan value and scan index + vice versa
            # Using exteded range as to be sure to take into account the case of only as single scan step
            self.toScanValue = lambda x: np.interp(x,np.arange(-0.5,len(self.scanStepExtended)-1),self.scanValuesExtended)
            self.fromScanValue = lambda x: np.interp(x,self.scanValuesExtended,np.arange(-0.5,len(self.scanStepExtended)-1))
        
            # Create additional axes
            self.secax_theta = self.ax_thetaIntegrated.secondary_yaxis('right', functions=(self.toScanValue, self.fromScanValue))
            self.secax_alpha = self.ax_alphaIntegrated.secondary_yaxis('right', functions=(self.toScanValue, self.fromScanValue))
            ylabel = scanLabel
            if not scanValueUnit is None:
                ylabel = ylabel+' ['+scanValueUnit+']'
            self.secax_theta.set_ylabel(ylabel)
            self.secax_alpha.set_ylabel(ylabel)
        
        
        ## Use scan values in stead of index
        def formatter(value): 
            return self.scanValueFormat.format(self.toScanValue(value),int(np.round(value)))
        self.indexSlider._format = formatter 

        if self.scanSteps !=1: # force a redraw of the slider
            self.indexSlider.set_val(0)
            #self.indexSlider.set_val(0)
        
        self.fig.canvas.mpl_connect('key_press_event',lambda event: self.onkeypress(event) )
        self.fig.canvas.mpl_connect('scroll_event',lambda event: self.onscroll(event))
        # Force plot of spectrum with index 0
        self.plotSpectrum(index=0)
        
        self.cid = self.fig.canvas.mpl_connect('button_press_event', lambda event: self.onclick(event,outputFunction=outputFunction))
        
        
        ## Create format coords for summed axes
        
        self.yformat = scanLabel+"={:.3f}"
        if not scanValueUnit is None:
            self.yformat = self.yformat +' ['+scanValueUnit+']'
        
        if len(self.xlabel.split(' [')) == 2:
            alphaSumfmt = ' = {:.3f} ['.join(self.xlabel.split(' ['))+ ' '
        else:
            alphaSumfmt = self.xlabel + ' = {:f} '

        if len(self.xlabel.split(' [')) == 2:
            twoThetaSumfmt = ' = {:.3f} ['.join(self.ylabel.split(' [')) + ' '
        else:
            twoThetaSumfmt = self.ylabel + ' = {:.3f} '
            
        self.format_coord_alphaSum = lambda x,y: (alphaSumfmt + self.yformat + ' scanStep={:.0f}' ).format(x,(self.toScanValue(y)),int(np.round(y)))
        self.format_coord_twoThetaSum = lambda x,y: (twoThetaSumfmt + self.yformat + ' scanStep={:.0f}').format(x,(self.toScanValue(y)),int(np.round(y)))

        self.format_coord_data = lambda x,y: (alphaSumfmt + twoThetaSumfmt).format(x,y)
        
        self.fig.tight_layout()
       
    def plotSpectrum(self,index=0,fromSlider=False):
        
        if self.index == index or index < 0 or index >= self.scanSteps:
            return
                    

        if not self.indexSlider.val == index: # If clicked on slider a float is provided. Reset to int
            self.indexSlider.set_val(index)
            return
        
        vmin,vmax = self.initialLimits
        #extent = np.array([[f(dat) for f in [np.nanmin,np.nanmax]] for dat in [ax.twoTheta[index],ax.pixelPosition[index]]]).flatten()
        
        if hasattr(self.ax_singleStep,'_pcolormesh'):
            self.ax_singleStep._pcolormesh.set_array(self.data[index].T)
            self.ax_singleStep.redraw_in_frame()
        else:
            self.ax_singleStep._pcolormesh = self.ax_singleStep.pcolormesh(self.twoThetaExtended,self.pixelPositionExtended,self.data[index].T,vmin=vmin,vmax=vmax,cmap=self.cmap)
            plt.draw()
    
        
        self.index = index
        if self.colorbar:
            if not hasattr(self,'_colorbar'):
                self.colorbarax = self.fig.add_subplot(self.gs[0, -1])
                self._colorbar = plt.colorbar(self.ax_singleStep._pcolormesh,cax=self.colorbarax)
                self._colorbar.set_label(self.dataLabel)
            
        
    
    def increaseAxis(self,step=1): # Call function to increase index
        index = self.index
        index+=step
        if index>=len(self.data):
            index = len(self.data)-1
        self.plotSpectrum(index)
    
    def decreaseAxis(self,step=1): # Call function to decrease index
        index = self.index
        index-=step
        if index<=-1:
            index = 0
        self.plotSpectrum(index)
        
        
    def onkeypress(self,event): # pragma: no cover
        if event.key in ['+','up']:
            self.increaseAxis()
        elif event.key in ['-','down']:
            self.decreaseAxis()
        elif event.key in ['home']:
            index = 0
            self.plotSpectrum(index)
        elif event.key in ['end']:
            index = len(self.data)-1
            self.plotSpectrum(index)
        elif event.key in ['pageup','ctrl++','ctrl+up']: # Pressing pageup or page down performs steps of 10
            self.increaseAxis(step=10)
        elif event.key in ['pagedown','ctrl+-','ctrl+down']:
            self.decreaseAxis(step=10)
            
    def sliders_on_changed(self,index):
        index = int(round(index))
        if index != self.index:
            self.plotSpectrum(index,fromSlider=True)
        
    def onscroll(self,event): # pragma: no cover
        if(event.button=='up'):
            self.increaseAxis()
        elif event.button=='down':
            self.decreaseAxis()
            
    def onclick(self,event,outputFunction=print):
        
        #outputFunction(event,event.inaxes)
        if not event.inaxes is None:
            self.event = event
            if event.inaxes in [self.ax_alphaIntegrated,self.ax_thetaIntegrated]:
                if event.button != 1: # not a left click
                    return
                # Check if anything but no special curser is chosen
                try:
                    C = self.fig.canvas.cursor().shape() # Only works for pyQt5 backend
                except:
                    pass
                else:
                    if C != 0:
                        return
                ## Plot corresponding scan step
                self.plotSpectrum(index = int(np.round(self.event.ydata)))
            elif event.inaxes == self.ax_singleStep:
                position = event.xdata,event.ydata
                outputFunction(event.inaxes.format_coord(*position))
            
        
        #### Functions to interface with format coord of axes with data
    def format_coord_alphaSum_getter(self):
        return self.ax_alphaIntegrated.format_coord
        
    def format_coord_alphaSum_setter(self,function):
        self.ax_alphaIntegrated.format_coord = function
        
    def format_coord_twoThetaSum_getter(self):
        return self.ax_thetaIntegrated.format_coord
        
    def format_coord_twoThetaSum_setter(self,function):
        self.ax_thetaIntegrated.format_coord = function
    
    def format_coord_data_getter(self):
        return self.ax_singleStep.format_coord
        
    def format_coord_data_setter(self,function):
        self.ax_singleStep.format_coord = function
        
    format_coord_alphaSum = property(format_coord_alphaSum_getter, format_coord_alphaSum_setter)
    format_coord_twoThetaSum = property(format_coord_twoThetaSum_getter, format_coord_twoThetaSum_setter)
    format_coord_data = property(format_coord_data_getter, format_coord_data_setter)
    

    def set_all_clim(self,vmin,vmax):
        for pc in [self.ax_alphaIntegrated,self.ax_thetaIntegrated,self.ax_singleStep]:
            pc._pcolormesh.set_clim(vmin,vmax)
    
    def set_clim(self,vmin,vmax):
        self.ax_singleStep._pcolormesh.set_clim(vmin,vmax)

    def set_clim_zIntegrated(self,vmin,vmax):
        self.ax_alphaIntegrated._pcolormesh.set_clim(vmin,vmax)

    def set_clim_thetaIntegrated(self,vmin,vmax):
        self.ax_thetaIntegrated._pcolormesh.set_clim(vmin,vmax)