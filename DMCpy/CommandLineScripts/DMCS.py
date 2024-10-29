from DMCpy import DataFile, _tools, DataSet
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# new update 17_june_2024


class MyGUI:
    def __init__(self, master):
        print('Welcome to DMCSpy!')       
         # Initialize GUI and create input variables
        self.master = master
        master.title("DMCSpy")

        # Variables for storing input values. We need to make the class have attributes which correspond to all of DMCpy inputs.
        # Those are: 
        # 1) Variables for obtaining the datafiles
        # 2) Variables related to the crystalline structure
        # 3) Variables for the HKL transformation
        # 4) Variables for 3D viewer

        
        # Variables for getting the DataFiles
        self.scanNumbers_var = tk.StringVar()
        self.dataFolder_var = tk.StringVar(value='/home/dmc/data/2024/')
        self.year_var = tk.StringVar(value='2024')
        self.Background_var = tk.StringVar()

        # Variables related for the unit cell
        self.a = tk.DoubleVar()
        self.b = tk.DoubleVar()
        self.c = tk.DoubleVar()
        self.alpha = tk.StringVar(value=90)
        self.beta = tk.StringVar(value=90)
        self.gamma = tk.StringVar(value=90)
        
        
        # Variables needed for HKL transformation
        self.Q1x = tk.DoubleVar(value=-4.126)
        self.Q2x = tk.DoubleVar(value=+2.166)
        self.Q1y = tk.DoubleVar(value=-2.240)
        self.Q2y = tk.DoubleVar(value=+0.114)
        self.Q1z = tk.DoubleVar(value=-0.285)
        self.Q2z = tk.DoubleVar(value=+0.165)

        self.H1 = tk.DoubleVar(value=1)
        self.H2 = tk.DoubleVar(value=0)
        self.K1 = tk.DoubleVar(value=0)
        self.K2 = tk.DoubleVar(value=1)
        self.L1 = tk.DoubleVar(value=0)
        self.L2 = tk.DoubleVar(value=0)

        # Variables needed for3D veiwer
        self.xbinsize = tk.StringVar(value=0.015)
        self.ybinsize = tk.StringVar(value=0.015)
        self.zbinsize = tk.StringVar(value=0.04)
        self.hkl = tk.IntVar() #toggles whether we use hkl or Q_x,y,z

    


        # Create input Widgets and Buttons - these are the things we actually interact with or labels on the GUI.
        tk.Label(master, text="Numors:").grid(row=1, column=0)
        tk.Entry(master, textvariable=self.scanNumbers_var).grid(row=1, column=1)

        tk.Label(master, text="Background:").grid(row=2, column=0)
        tk.Entry(master, textvariable=self.Background_var).grid(row=2, column=1)

        tk.Label(master, text="Data Folder:").grid(row=3, column=0)
        tk.Entry(master, textvariable=self.dataFolder_var).grid(row=3, column=1)
        tk.Button(master, text="Browse", command=self.browse_data_folder).grid(row=3, column=2)

        tk.Button(master, text="Load Project", command=self.Load_settings).grid(row=1, column=2)
        tk.Button(master, text="Save Project", command=self.Save_settings).grid(row=2, column=2)


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

        tk.Label(master, text='Coordinate Transformation').grid(row=5, column=6)
        tk.Label(master, text='Q1x').grid(row=6, column=5)
        tk.Label(master, text='Q1y').grid(row=6, column=6)
        tk.Label(master, text='Q1z').grid(row=6, column=7)

        tk.Entry(master, textvariable=self.Q1x).grid(row=7, column=5)
        tk.Entry(master, textvariable=self.Q1y).grid(row=7, column=6)
        tk.Entry(master, textvariable=self.Q1z).grid(row=7, column=7)

        tk.Label(master, text='Q2x').grid(row=8, column=5)
        tk.Label(master, text='Q2y').grid(row=8, column=6)
        tk.Label(master, text='Q2z').grid(row=8, column=7)

        tk.Entry(master, textvariable=self.Q2x).grid(row=9, column=5)
        tk.Entry(master, textvariable=self.Q2y).grid(row=9, column=6)
        tk.Entry(master, textvariable=self.Q2z).grid(row=9, column=7)

        tk.Label(master, text='H1').grid(row=10, column=5)
        tk.Label(master, text='K1').grid(row=10, column=6)
        tk.Label(master, text='L1').grid(row=10, column=7)

        tk.Entry(master, textvariable=self.H1).grid(row=11, column=5)
        tk.Entry(master, textvariable=self.K1).grid(row=11, column=6)
        tk.Entry(master, textvariable=self.L1).grid(row=11, column=7)

        tk.Label(master, text='H2').grid(row=12, column=5)
        tk.Label(master, text='K2').grid(row=12, column=6)
        tk.Label(master, text='L2').grid(row=12, column=7)

        tk.Entry(master, textvariable=self.H2).grid(row=13, column=5)
        tk.Entry(master, textvariable=self.K2).grid(row=13, column=6)
        tk.Entry(master, textvariable=self.L2).grid(row=13, column=7)

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


        # Advanced functons: 
        # Export data 3D | Plot plane 2D | Plot Cut 1D

        # Buttons for the Adv. Functions
        tk.Button(master, text="Export 3D Data",
                  command=self.Export_3D_data).grid(row=13, column=0)
        tk.Button(master, text="Cut 2D Plane",
                  command=self.Cut_2D_Plane).grid(row=13, column=1)
        tk.Button(master, text="Cut 1D Line",
                  command=self.Cut_1D_Line).grid(row=13, column=2)
        
        # Variables for the Adv. Functions

    def Export_3D_data(self): # These functions below tell us what to do when we click a button.
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

    def Save_settings(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".npy", filetypes=[("Text files", "*.npy"), ("All files", "*.*")])
        if file_path:
            value_dict = self.Get_values()
            np.save(file_path, value_dict, allow_pickle=True)
            print('Project saved! ')
        else:
            print('No savefile given')
            

    def Get_values(self):
        value_dict = {}
        attribs = vars(self)
        for i in attribs:
            try: 
                value_dict[str(i)] = attribs[i].get()
            except:
                if i != 'master':
                    print('Error with attribute: ', i)
        return(value_dict)

    def Load_settings(self):
        file_path = filedialog.askopenfilename(defaultextension=".npy", filetypes=[("Text files", "*.npy"), ("All files", "*.*")])
        new_vals = np.load(file_path, allow_pickle=True).tolist()
        attribs = vars(self)
        for i in attribs:
            if i != 'master':
                try:
                    attribs[i].set(new_vals[i])
                except Exception as error:
                    print('error with attribute: ', str(i))
                    print(error)
                    print()
        print('loaded')

    def Set_values(self, newvals):
        print('I try set values :)')

    def Data3D_Export(self):

        print('Exporting the 3D Data')

        self.intialize_transformation()

        print('Binning the data... Plase be patient! (:\n')
    
        intensities,bins,errors = self.ds.binData3D(dqx=self.Binsize3Dx.get(),
                                                    dqy=self.Binsize3Dy.get(),
                                                    dqz=self.Binsize3Dz.get(),
                                                    rlu=self.hkl.get())
        
        np.save(self.scanNumbers_var.get()+'_Cut3D_Intensities.npy', intensities)
        np.save(self.scanNumbers_var.get()+'_Cut3D_BinEdges.npy', bins)
        np.save(self.scanNumbers_var.get()+'_Cut3D_Errors.npy', errors)
        print('Cut3D Saved\n!')
        
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
        print('Binning the data... Plase be patient! (:\n')

        self.intialize_transformation()

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
        print('Binning the data... Plase be patient! (:\n')

        self.intialize_transformation()

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
        print(f"Closed Window\n")

        # Close the settings window
        self.settings_window.destroy()

    def on_click_hkl_message(self):
        print('\nIf Checked:')
        print('Please Ensure Coordinate Transformation and Unit Cell Tables has been filled')
        print('Remember to update Unit Cell to DataFile via Get it!')

    def initialise_data(self):
        # Get input values from the GUI
        scanNumbers = self.scanNumbers_var.get()
        dataFolder = self.dataFolder_var.get()
        year = int(self.year_var.get())

        a = self.a.get()
        b = self.b.get()
        c = self.c.get()
        al = self.alpha.get()
        be = self.beta.get()
        ga = self.gamma.get()
        filePath = _tools.fileListGenerator(scanNumbers,
                                            dataFolder,
                                            year=year)

        if a == '':
            print('\nNo Unit Cell Information, continuing generically...\n')
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
            print('Unit Cell:', unitCell)

        dataFiles = [DataFile.loadDataFile(
            dFP, unitCell=unitCell) for dFP in filePath]
        self.ds = DataSet.DataSet(dataFiles)

        if self.Background_var.get() != '':
            print('Performing background subtraction... Be Patient! (: ')
            filePath_sub =  _tools.fileListGenerator(self.Background_var.get(),
                                            dataFolder,
                                            year=year)
            dataFiles_sub = [DataFile.loadDataFile(
            dFP, unitCell=unitCell) for dFP in filePath_sub]
            ds_sub = DataSet.DataSet(dataFiles_sub)
            self.ds.directSubtractDS(ds_sub,saveToFile=True,
                                     saveToNewFile='subtracted_data.hdf')

        print('\nInitalised {} DataFiles\n'.format(len(self.ds)))

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
        
        self.intialize_transformation()

    def Cut2D_Export(self):
        print('Plotting a 2D Plane')
        print('Binning the data... Plase be patient! (:\n')

        self.Cut2D_cmin = tk.DoubleVar(value=0)
        self.Cut2D_cmax = tk.DoubleVar(value=0.0001)

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
        print('Cut2D Saved\n!')
        plt.close()

    def Cut2D_Plot(self):

        print('Plotting a 2D Plane')
        print('Binning the data... Plase be patient! (:\n')

        self.Cut2D_cmin = tk.DoubleVar(value=0)
        self.Cut2D_cmax = tk.DoubleVar(value=0.0001)

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
        print('\nBinning the data... Please be patient (: \n ')
        use_hkl = self.hkl.get()
        if use_hkl == 1:
            print('HKL Transform requested.. \n')
            self.intialize_transformation()
        

        print(float(self.xbinsize.get()))
        print(float(self.ybinsize.get()))
        print(float(self.zbinsize.get()))
        Viewer = self.ds.Viewer3D(float(self.xbinsize.get()),
                                  float(self.ybinsize.get()),
                                  float(self.zbinsize.get()),
                                  rlu=use_hkl,
                                  steps=150)
        
        
        Viewer.ax.axis(self.Viewer_3D_axes.get())
        Viewer.set_clim(self.Viewer_3D_colur_min.get(),
                        self.Viewer_3D_colur_max.get())
        plt.show()

    def intialize_transformation(self):
        self.q1 = [float(self.Q1x.get()), 
                   float(self.Q1y.get()), 
                   float(self.Q1z.get())]
        self.q2 = [float(self.Q2x.get()), 
                   float(self.Q2y.get()), 
                   float(self.Q2z.get())]
        self.HKL1 = [int(self.H1.get()), 
                     int(self.K1.get()), 
                     int(self.L1.get())]
        self.HKL2 = [int(self.H2.get()), 
                    int(self.K2.get()), 
                     int(self.L2.get())]

        print(self.q1, self.q2, self.HKL1, self.HKL2)
        self.ds.alignToRefs(q1=self.q1, q2=self.q2,
                            HKL1=self.HKL1, HKL2=self.HKL2)

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


def main():
    root = tk.Tk()
    gui = MyGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
