#import sys
from DMCpyZEBRA import DataFile, _tools, DataSet
#import json
#import os
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import sys
import warnings


#try:
#    import IPython
#    shell = IPython.get_ipython()
#    shell.enable_matplotlib(gui='qt')
#except ImportError:
#    pass


class MyGUI:
    def __init__(self, master):       
         # Initialize GUI and create input variables
        self.master = master
        master.title("ZEBRASpy")

                # Variables for getting the DataFiles
        self.scanNumbers_var = tk.StringVar()
        self.dataFolder_var = tk.StringVar()
        self.year_var = tk.StringVar(value='2026')

                # Variables needed for the unit cell
        self.a = tk.DoubleVar(value=12.626)
        self.b = tk.DoubleVar(value=6.199)
        self.c = tk.DoubleVar(value=5.923)
        self.alpha = tk.StringVar(value=90)
        self.beta = tk.StringVar(value=90)
        self.gamma = tk.StringVar(value=90)

                # Variables needed for3D veiwer
        self.xbinsize = tk.StringVar(value=0.015)
        self.ybinsize = tk.StringVar(value=0.015)
        self.zbinsize = tk.StringVar(value=0.04)
        self.hkl = tk.IntVar() #toggles whether we use hkl or Q_x,y,z

        self.Background_var = tk.StringVar()

        # Create input Widgets and Buttons


        tk.Label(master, text="Numors:").grid(row=1, column=0)
        tk.Entry(master, textvariable=self.scanNumbers_var).grid(row=1, column=1)

        tk.Label(master, text="Background:").grid(row=2, column=0)
        tk.Entry(master, textvariable=self.Background_var).grid(row=2, column=1)

        tk.Label(master, text="Data Folder:").grid(row=3, column=0)
        tk.Entry(master, textvariable=self.dataFolder_var).grid(row=3, column=1)
        tk.Button(master, text="Browse", command=self.browse_data_folder).grid(row=3, column=2)

        tk.Label(master, text="Year:").grid(row=4, column=0)
        tk.Entry(master, textvariable=self.year_var).grid(row=4, column=1)

        tk.Button(master, text="Get it!", command=self.initialise_data).grid(row=4, column=2)
        tk.Button(master, text="Interactive Viewer", command=self.plot_interactive_viewer).grid(
                          row=6, column=1, columnspan=1, rowspan=2)

        tk.Button(master, text="3D Viewer", command=self.plot_3D_viewer).grid(row=8, column=1,
                                                                              columnspan=1, rowspan=2)

        tk.Checkbutton(master, text='HKL On', variable=self.hkl, onvalue=1, offvalue=0,
                       command=self.on_click_hkl_message).grid(row=9, column=2)

        tk.Button(master, text="Config", command=self.open_settings_window_IV).grid(row=6, column=0,
                                                                             columnspan=1, rowspan=2)
        tk.Button(master, text="Config", command=self.open_settings_window_3D).grid(row=8, column=0,
                                                                             columnspan=1, rowspan=2)


        tk.Label(master, text='x_bin size').grid(row=10, column=0)
        tk.Label(master, text='y_bin size').grid(row=10, column=1)
        tk.Label(master, text='z_bin size').grid(row=10, column=2)

        tk.Entry(master, textvariable=self.xbinsize).grid(row=11, column=0)
        tk.Entry(master, textvariable=self.ybinsize).grid(row=11, column=1)
        tk.Entry(master, textvariable=self.zbinsize).grid(row=11, column=2)

        tk.Label(master, text='Unit Cell').grid(row=0, column=6)
        tk.Label(master, text='a').grid(row=1, column=4)
        tk.Label(master, text='b').grid(row=2, column=4)
        tk.Label(master, text='c').grid(row=3, column=4)

        tk.Label(master, text='    al').grid(row=1, column=6)
        tk.Label(master, text='    be').grid(row=2, column=6)
        tk.Label(master, text='    ga').grid(row=3, column=6)

        tk.Entry(master, textvariable=self.a).grid(row=1, column=5)
        tk.Entry(master, textvariable=self.b).grid(row=2, column=5)
        tk.Entry(master, textvariable=self.c).grid(row=3, column=5)
        tk.Entry(master, textvariable=self.alpha).grid(row=1, column=7)
        tk.Entry(master, textvariable=self.beta).grid(row=2, column=7)
        tk.Entry(master, textvariable=self.gamma).grid(row=3, column=7)

        tk.Label(master, text='').grid(row=4, column=6)


        tk.Button(master, text="Overwrite unit cell", command=self.overwrite_unit_cell).grid(row=5, column=6,columnspan=1, rowspan=2)
        tk.Button(master, text="Load UB", command=self.load_ub).grid(row=7, column=6,columnspan=1, rowspan=2)



        # Text widget
        self.text_box = tk.Text(master, height=8, wrap='word')
        self.text_box.grid(row=14, column=0, columnspan=8, padx=10, pady=(10,10), sticky='nsew')

        # Optional: color stderr in red
        self.text_box.tag_configure('stderr', foreground='red')

        # Redirect stdout and stderr
        self.stdout_redirector = TextRedirector(self.text_box, 'stdout')
        sys.stdout = self.stdout_redirector
        sys.stderr = self.stdout_redirector


        # Optional: redirect warnings
        def showwarning(message, category, filename, lineno, file=None, line=None):
            print(f"Warning: {message}", file=sys.stderr)

        warnings.showwarning = showwarning


        # Variables for storing colorbar limits in Interactive Viewer
        self.IV_colorbar_min_var = tk.DoubleVar(value=0)  # Default values
        self.IV_colorbar_max_var = tk.DoubleVar(value=2) 
        self.IV_int_colorbar_min_var = tk.DoubleVar(value=0) 
        self.IV_int_colorbar_max_var = tk.DoubleVar(value=100)  

        # Variable for the 3D Viewer
        self.Viewer_3D_axes = tk.StringVar(value='equal')
        self.Viewer_3D_colur_min = tk.DoubleVar(value = 0)
        self.Viewer_3D_colur_max = tk.DoubleVar(value = 0.005)
        self.Viewer_3D_axis_1 = tk.StringVar(value = '100')
        self.Viewer_3D_axis_2 = tk.StringVar(value ='010')


        # Advances functons: 
        # Export data 3D | Plot plane 2D | Plot Cut 1D

        # Buttons for the Adv. Functions
        tk.Button(master, text="Export 3D Data",
                  command=self.Export_3D_data).grid(row=13, column=0)
        tk.Button(master, text="Cut 2D Plane",
                  command=self.Cut_2D_Plane).grid(row=13, column=1)
        tk.Button(master, text="Cut 1D Line",
                  command=self.Cut_1D_Line).grid(row=13, column=2)
        
        # Variables for the Adv. Functions



    def Export_3D_data(self):
        # Create a new window for settings
        self.settings_window = tk.Toplevel(self.master)
        self.settings_window.title("Export 3D Data Settings")
    
        # Creating variables for the 3D binning
        self.Binsize3Dx = tk.DoubleVar(value=0.01)
        self.Binsize3Dy = tk.DoubleVar(value=0.01)
        self.Binsize3Dz = tk.DoubleVar(value=0.01)


        # Entry widgets for bin sizes limits
        tk.Label(self.settings_window, text="x_bin size").grid(row=0, column=0)
        tk.Entry(self.settings_window, textvariable=self.Binsize3Dx).grid(row=1, column=0)

        tk.Label(self.settings_window, text="y_bin size").grid(row=0, column=1)
        tk.Entry(self.settings_window, textvariable=self.Binsize3Dy).grid(row=1, column=1)

        tk.Label(self.settings_window, text="z_bin size").grid(row=0, column=2)
        tk.Entry(self.settings_window, textvariable=self.Binsize3Dz).grid(row=1, column=2)

        tk.Button(self.settings_window, text="Export Data",
                   command=self.Data3D_Export).grid(row=3, column=0)
        tk.Button(self.settings_window, text="Close",
                   command=self.apply_settings).grid(row=3, column=2)

    def Data3D_Export(self):

        print('Exporting the 3D Data')


        print('Binning the data... Plase be patient! (:')
    
        intensities,bins,errors = self.ds.binData3D(dqx=self.Binsize3Dx.get(),
                                                    dqy=self.Binsize3Dy.get(),
                                                    dqz=self.Binsize3Dz.get(),
                                                    rlu=self.hkl.get())
        
        np.save(self.scanNumbers_var.get()+'_Cut3D_Intensities.npy', intensities)
        np.save(self.scanNumbers_var.get()+'_Cut3D_BinEdges.npy', bins)
        np.save(self.scanNumbers_var.get()+'_Cut3D_Errors.npy', errors)
        print('Cut3D Saved!')
        


    def Cut_1D_Line(self):
        # Create a new window for 1D LineCut Settings

        self.settings_window = tk.Toplevel(self.master)
        self.settings_window.title("1D LineCut Settings")

        # Defining new variables required for cutting a 2D plane
        self.Cut1D_Start = tk.StringVar(value='0.0,0.0,0.0')
        self.Cut1D_End = tk.StringVar(value='1.0,0.0,0.0')

        self.Cut1D_Wperp = tk.DoubleVar(value=0.1)
        self.Cut1D_W_Z = tk.DoubleVar(value=0.1)
        self.Cut1D_Step = tk.DoubleVar(value=0.005)

        # Entry widgets for colorbar limits
        tk.Label(self.settings_window, text="Insert Start and End Points to Cut").grid(
            row=0, column=0, columnspan=4)

        tk.Label(self.settings_window, text="Start:").grid(row=1, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut1D_Start).grid(
            row=1, column=1, columnspan=2)

        tk.Label(self.settings_window, text="End:").grid(row=2, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut1D_End).grid(
            row=2, column=1, columnspan=2)

        tk.Label(self.settings_window, text="KWargs for the 1D Cut").grid(
            row=3, column=0, columnspan=2)
        
        tk.Label(self.settings_window, text="Width along Qz /AA^-1 :").grid(row=4, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut1D_W_Z).grid(
            row=4, column=1, columnspan=2)
        
        tk.Label(self.settings_window, text="Width Perp /AA^-1 :").grid(row=5, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut1D_Wperp).grid(
            row=5, column=1, columnspan=2)

        tk.Label(self.settings_window, text="Stepsize /HKL On :").grid(row=6, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut1D_Step).grid(
            row=6, column=1, columnspan=2)
        
        # Apply button to update settings
        tk.Button(self.settings_window, text="Export 1D Data",
                   command=self.Cut1D_Export).grid(row=8, column=1, columnspan=1)
        tk.Button(self.settings_window, text="Plot 1D Cut",
                   command=self.Cut1D_Plot).grid(row=8, column=0, columnspan=1)
        tk.Button(self.settings_window, text="Close",
                   command=self.apply_settings).grid(row=8, column=2, columnspan=1)
        
    def Cut1D_Export(self):
        print('Exporting the 1D Plane')
        print('Binning the data... Plase be patient! (:')


        p1 = [float(x) for x in self.Cut1D_Start.get().split(',')]
        p2 = [float(x) for x in self.Cut1D_End.get().split(',')]

        kwargs = {
                'width' : self.Cut1D_Wperp.get(),
                'widthZ' : self.Cut1D_W_Z.get(),
                'stepSize' : self.Cut1D_Step.get(),
                'rlu' : self.hkl.get(),
                'optimize' : False,
                'marker' : 'o',
                'color' : 'green',
                'markersize' : 8,
                'mew' : 1.5,
                'linewidth' : 1.5,
                'capsize' : 3,
                'linestyle' : (0, (1, 1)),
                'mfc' : 'white',
                }

        positionVector,I,err,ax = self.ds.plotCut1D(p1, p2, **kwargs)
        np.save(self.scanNumbers_var.get()+'_Cut1D_Position.npy', positionVector)
        np.save(self.scanNumbers_var.get()+'_Cut1D_Intensities.npy', I)
        np.save(self.scanNumbers_var.get()+'_Cut1D_Errors.npy', err)
        print("Saved!")


    def Cut1D_Plot(self):
        print('Exporting the 1D Plane')
        print('Binning the data... Plase be patient! (:')


        p1 = [float(x) for x in self.Cut1D_Start.get().split(',')]
        p2 = [float(x) for x in self.Cut1D_End.get().split(',')]

        kwargs = {
                'width' : self.Cut1D_Wperp.get(),
                'widthZ' : self.Cut1D_W_Z.get(),
                'stepSize' : self.Cut1D_Step.get(),
                'rlu' : self.hkl.get(),
                'optimize' : False,
                'marker' : 'o',
                'color' : 'green',
                'markersize' : 8,
                'mew' : 1.5,
                'linewidth' : 1.5,
                'capsize' : 3,
                'linestyle' : (0, (1, 1)),
                'mfc' : 'white',
                }

        positionVector,I,err,ax = self.ds.plotCut1D(p1, p2, **kwargs)
        fig = ax.get_figure()
        plt.show()

    def open_settings_window_IV(self):
        # Create a new window for settings
        self.settings_window = tk.Toplevel(self.master)
        self.settings_window.title("Interactive Viewer Settings")

        # Entry widgets for colorbar limits
        tk.Label(self.settings_window, text="Raw Colorbar Min:").grid(row=0, column=0)
        tk.Entry(self.settings_window, textvariable=self.IV_colorbar_min_var).grid(row=0, column=1)

        tk.Label(self.settings_window, text="Raw Colorbar Max:").grid(row=1, column=0)
        tk.Entry(self.settings_window, textvariable=self.IV_colorbar_max_var).grid(row=1, column=1)

        tk.Label(self.settings_window, text="Int. Colorbar Min:").grid(row=2, column=0)
        tk.Entry(self.settings_window, textvariable=self.IV_int_colorbar_min_var).grid(row=2, column=1)

        tk.Label(self.settings_window, text="Int. Colorbar Max:").grid(row=3, column=0)
        tk.Entry(self.settings_window, textvariable=self.IV_int_colorbar_max_var).grid(row=3, column=1)

        # Apply button to update settings
        tk.Button(self.settings_window, text="Close", command=self.apply_settings).grid(row=4, column=0, columnspan=2)

    def open_settings_window_3D(self):
        # Create a new window for settings
        self.settings_window = tk.Toplevel(self.master)
        self.settings_window.title("3D Viewer Settings")

        # Entry widgets for colorbar limits
        tk.Label(self.settings_window, text="3D Colorbar Min:").grid(row=0, column=0)
        tk.Entry(self.settings_window, textvariable=self.Viewer_3D_colur_min).grid(row=0, column=1)

        tk.Label(self.settings_window, text="3D Colorbar Min:").grid(row=1, column=0)
        tk.Entry(self.settings_window, textvariable=self.Viewer_3D_colur_max).grid(row=1, column=1)

        tk.Label(self.settings_window, text="Axes Aspect:").grid(row=2, column=0)
        tk.Entry(self.settings_window, textvariable=self.Viewer_3D_axes).grid(row=2, column=1)

        tk.Label(self.settings_window, text="Axis 1:").grid(row=3, column=0)
        tk.Entry(self.settings_window, textvariable=self.Viewer_3D_axis_1).grid(row=3, column=1)


        tk.Label(self.settings_window, text="Axis 2:").grid(row=4, column=0)
        tk.Entry(self.settings_window, textvariable=self.Viewer_3D_axis_2).grid(row=4, column=1)


        # Apply button to update settings
        tk.Button(self.settings_window, text="Close",
                   command=self.apply_settings).grid(row=5, column=0, columnspan=1)
        # Apply button to update settings
        tk.Button(self.settings_window, text="Set New Projection",
                   command=self.set_new_3D_projection).grid(row=5, column=1, columnspan=1)

    def set_new_3D_projection(self):
        # checks if the values have been changed and then either pdates or doesn't. Closes regardless.
        if self.Viewer_3D_axis_2.get() == '010' and self.Viewer_3D_axis_1.get() == '100':
            print('Keeping standard axes')
            self.apply_settings()
        else:
            print('Updating the 3D Viewer with new projection vectors')
            p1 = np.array(list(self.Viewer_3D_axis_1.get() ), dtype=int)
            p2 = np.array(list(self.Viewer_3D_axis_2.get() ), dtype=int)
            self.ds.setProjectionVectors(p1,p2,p3=None)
            self.apply_settings()

    def apply_settings(self):
        # Retrieve values from the settings window and apply them
        print(f"Closed Window")

        # Close the settings window
        self.settings_window.destroy()

    def on_click_hkl_message(self):
        if self.hkl==True:
            print('Using UB from HDF file')

    def load_unit_cell(self):
        a = self.a.get()
        b = self.b.get()
        c = self.c.get()
        al = self.alpha.get()
        be = self.beta.get()
        ga = self.gamma.get()

        if a == '':
            print('No Unit Cell Information, continuing generically...')
            unitCell = np.array([1,
                                 1,
                                 1,
                                 1,
                                 1,
                                 1])
            self.unitcelled = False
        else:
            unitCell = np.array([
                                float(a),
                                float(b),
                                float(c),
                                float(al),
                                float(be),
                                float(ga)])
            self.unitcelled = True
        return unitCell


    def initialise_data(self):
        # Get input values from the GUI
        scanNumbers = self.scanNumbers_var.get()
        dataFolder = self.dataFolder_var.get()
        year = int(self.year_var.get())

        filePath = _tools.fileListGenerator(scanNumbers,
                                            dataFolder,
                                            year=year,
                                            instrument='zebra')


        print('Ignoring entered unit cell and using values from hdf file...')
        unitCell = None
        dataFiles = [DataFile.loadDataFile(
            dFP, unitCell=unitCell) for dFP in filePath]
        self.ds = DataSet.DataSet(dataFiles)
        print(f'Unit Cell: {self.ds[0].sample.unitCell}')

        if self.Background_var.get() != '':
            print('Performing background subtraction... Be Patient! (: ')
            filePath_sub =  _tools.fileListGenerator(self.Background_var.get(),
                                            dataFolder,
                                            year=year,
                                            instrument='zebra')
            dataFiles_sub = [DataFile.loadDataFile(
            dFP, unitCell=unitCell) for dFP in filePath_sub]
            ds_sub = DataSet.DataSet(dataFiles_sub)
            self.ds.directSubtractDS(ds_sub,saveToFile=True,
                                     saveToNewFile='subtracted_data.hdf')

        print('Initalised {} DataFiles'.format(len(self.ds)))

    def overwrite_unit_cell(self):
        ub = self.ds[0].sample.UB
        unitCell = self.load_unit_cell()
        for df in self.ds:
            df.sample.unitCell = unitCell
            df.sample.updateCell()

    def browse_data_folder(self):
        folder_path = filedialog.askdirectory()
        self.dataFolder_var.set(folder_path)

    def Cut_2D_Plane(self):
        # Create a new window for 2D Plane settings

        self.settings_window = tk.Toplevel(self.master)
        self.settings_window.title("2D Cut Settings")

        # Defining new variables required for cutting a 2D plane
        self.Cut2D_plane_point1 = tk.StringVar(value='0.0,0.0,0.0')
        self.Cut2D_plane_point2 = tk.StringVar(value='1.0,0.0,0.0')
        self.Cut2D_plane_point3 = tk.StringVar(value='0.0,1.0,0.0')

        self.Cut2D_width = tk.DoubleVar(value=0.1)
        self.Cut2D_dQx = tk.DoubleVar(value=0.005)
        self.Cut2D_dQy = tk.DoubleVar(value=0.005)

        self.Cut2D_cmin = tk.DoubleVar(value=0)
        self.Cut2D_cmax = tk.DoubleVar(value=0.0001)

        # Entry widgets for colorbar limits
        tk.Label(self.settings_window, text="Insert three points to define the plane").grid(
            row=0, column=0, columnspan=4)

        tk.Label(self.settings_window, text="Point 1:").grid(row=1, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut2D_plane_point1).grid(
            row=1, column=1, columnspan=2)

        tk.Label(self.settings_window, text="Point 2:").grid(row=2, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut2D_plane_point2).grid(
            row=2, column=1, columnspan=2)

        tk.Label(self.settings_window, text="Point 3:").grid(row=3, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut2D_plane_point3).grid(
            row=3, column=1, columnspan=2)

        tk.Label(self.settings_window, text="KWargs for Cut").grid(
            row=4, column=0, columnspan=4)
        
        tk.Label(self.settings_window, text="Width /AA^-1 :").grid(row=5, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut2D_width).grid(
            row=5, column=1, columnspan=2)

        tk.Label(self.settings_window, text="dQx /HKL On :").grid(row=6, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut2D_dQx).grid(
            row=6, column=1, columnspan=2)

        tk.Label(self.settings_window, text="dQy /HKL On :").grid(row=7, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut2D_dQy).grid(
            row=7, column=1, columnspan=2)
        
        tk.Label(self.settings_window, text="Cbar Min :").grid(row=8, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut2D_cmin).grid(
            row=8, column=1, columnspan=2)
        
        tk.Label(self.settings_window, text="Cbar Max :").grid(row=9, column=0)
        tk.Entry(self.settings_window, textvariable=self.Cut2D_cmax).grid(
            row=9, column=1, columnspan=2)

        # Apply button to update settings
        tk.Button(self.settings_window, text="Export 2D Data",
                   command=self.Cut2D_Export).grid(row=10, column=1, columnspan=1)
        tk.Button(self.settings_window, text="Plot 2D Plane",
                   command=self.Cut2D_Plot).grid(row=10, column=0, columnspan=1)
        tk.Button(self.settings_window, text="Close",
                   command=self.apply_settings).grid(row=10, column=2, columnspan=1)
        

    def Cut2D_Export(self):
        print('Plotting a 2D Plane')
        print('Binning the data... Plase be patient! (:')

        # these points define the plane that will be cut
        # the in-line for loops are a nice way of conveting the string inputs into a list of floats
        p1 = [float(x) for x in self.Cut2D_plane_point1.get().split(',')]
        p2 = [float(x) for x in self.Cut2D_plane_point2.get().split(',')]
        p3 = [float(x) for x in self.Cut2D_plane_point3.get().split(',')]
        
        points = np.array([p1, p2, p3])

        kwargs = {
        'dQx' : self.Cut2D_dQx.get(),
        'dQy' : self.Cut2D_dQy.get(),
        'steps' : 151,
        'rlu' : self.hkl.get(),
        'rmcFile' : True,
        'colorbar' : True,
        }

        ax,returndata,bins = self.ds.plotQPlane(points=points,
                                                width=self.Cut2D_width.get(),
                                                **kwargs)
        np.save(self.scanNumbers_var.get()+'_Cut2D_Data.npy', returndata)
        np.save(self.scanNumbers_var.get()+'_Cut2D_BinEdges.npy', bins)
        print('Cut2D Saved!')
        plt.close()

    def Cut2D_Plot(self):
        print('Plotting a 2D Plane')
        print('Binning the data... Plase be patient! (:')

        # these points define the plane that will be cut
        # the in-line for loops are a nice way of conveting the string inputs into a list of floats
        p1 = [float(x) for x in self.Cut2D_plane_point1.get().split(',')]
        p2 = [float(x) for x in self.Cut2D_plane_point2.get().split(',')]
        p3 = [float(x) for x in self.Cut2D_plane_point3.get().split(',')]
        
        points = np.array([p1, p2, p3])

        kwargs = {
        'dQx' : self.Cut2D_dQx.get(),
        'dQy' : self.Cut2D_dQy.get(),
        'steps' : 151,
        'rlu' : self.hkl.get(),
        'rmcFile' : True,
        'colorbar' : True,
        }

        ax,returndata,bins = self.ds.plotQPlane(points=points,
                                                width=self.Cut2D_width.get(),
                                                **kwargs)
        ax.set_clim(self.Cut2D_cmin.get(), self.Cut2D_cmax.get())
        print(self.Cut2D_cmax.get())
        plt.show()

    def plot_interactive_viewer(self):
        for i in range(len(self.ds)):
            print(i)
            IA = self.ds[i].InteractiveViewer()
            IA.set_clim(self.IV_colorbar_min_var.get(), self.IV_colorbar_max_var.get())
            IA.set_clim_zIntegrated(self.IV_int_colorbar_min_var.get(),
                                    self.IV_int_colorbar_max_var.get())
            plt.title(i)
            plt.show()

    def plot_3D_viewer(self):
        print('Binning the data... Please be patient (:  ')
        use_hkl = self.hkl.get()
        if use_hkl == 1:
            print('HKL Transform requested.. ')
        
        print('X Y Z bin sizes: {}, {}, {}'.format(float(self.xbinsize.get()),float(self.ybinsize.get()),float(self.zbinsize.get())))
        steps = None 

        Viewer = self.ds.Viewer3D(float(self.xbinsize.get()),
                                  float(self.ybinsize.get()),
                                  float(self.zbinsize.get()),
                                  rlu=use_hkl,
                                  steps=steps)
        
        
        Viewer.ax.axis(self.Viewer_3D_axes.get())
        Viewer.set_clim(self.Viewer_3D_colur_min.get(),
                        self.Viewer_3D_colur_max.get())
        plt.show()

    

    def load_ub(self):
        ub_fname = filedialog.askopenfilename(title='Select UB matrix')
        ub = []
        with open(ub_fname,'r') as file:
            for line in file:
                l = line.strip().translate(str.maketrans("[],;", "    "))
                for arg in l.split():
                    try:
                        ub.append(float(arg))
                    except ValueError:
                        pass
        if len(ub)!=9:
            raise ValueError(f'Read UB matrix does not contain the correct number of entries {ub}')

        ub = np.array(ub)

        print(self.ds[0].sample.UB/(2*np.pi))
        for df in self.ds:
            df.sample.UB = -2*np.pi*ub.reshape(3,3)
            df.sample.getProjectionVectorsFromUB()
        print(f'Replaced UB with: \n {ub.reshape(3,3)}')
 

    def run_script(self):
        # Insert the rest of your script here

        width = 0.1
        points = np.array([[-0.0, 0.0, 0.0],
                           [-0.0, 0.0, 1.0],
                           [-0.0, 1.0, 0.0]])

        kwargs = {
            'dQx': 0.005,
            'dQy': 0.005,
            'steps': 151,
            'rlu': True,
            'rmcFile': True,
            'colorbar': True,
        }

        ax, returndata, bins = ds.plotQPlane(
            points=points, width=width, **kwargs)
        ax.set_clim(0, 0.0001)
        plt.show()

    def on_closing(self):
        sys.stdout =  sys.__stdout__
        sys.stderr =  sys.__stderr__
        self.stdout_redirector.flush_to_terminal()
        self.master.destroy()
        


class TextRedirector:
    def __init__(self, text_box, tag='stdout'):
        self.text_box= text_box
        self.tag = tag

    def write(self,*args, sep=' ', end='\n'):
    
        timestamp = datetime.now().strftime("[%H:%M:%S]")
        message = sep.join(str(arg) for arg in args).rstrip('\n')+end 

        if not message.strip():
            return
        
        # Enable textbox temporarily
        self.text_box.config(state='normal')
        self.text_box.insert('end', f"{timestamp} {message}")
        self.text_box.see('end')  # Scroll to the bottom
        self.text_box.config(state='disabled')
        self.text_box.update_idletasks()

    def flush(self):
        pass  # Needed for compatibility with file-like behavior

    def flush_to_terminal(self):
        try:
            self.text_box.configure(state='normal')
            content = self.text_box.get('1.0', 'end').strip()
            self.text_box.configure(state='disabled')
            if content:
                print("\n--- Output from GUI Text Box ---", file=sys.__stdout__)
                print(content, file=sys.__stdout__)
                print("--- End of Output ---\n", file=sys.__stdout__)
        except Exception as e:
            print(f"Failed to flush TextRedirector contents: {e}", file=sys.__stderr__)

def main():
    root = tk.Tk()
    gui = MyGUI(root)
    root.protocol("WM_DELETE_WINDOW",gui.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
