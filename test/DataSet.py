from xml.dom.minidom import Attr

from DMCpy import DataSet
from DMCpy import DataFile
import os.path
import numpy as np
import matplotlib
matplotlib.use('tkagg')


def test_init():
    ds = DataSet.DataSet()

    assert(len(ds)==0)

    df = DataFile.loadDataFile(os.path.join('data','dmc2021n{:06d}.hdf'.format(494)))

    ds2 = DataSet.DataSet([df])
    assert(len(ds2)==1)

    ds3 = DataSet.DataSet(dataFiles=df)

    print(ds3.__dict__)
    print(ds2.__dict__)

    assert(ds2==ds3)


def test_load():

    fileNumbers = [494,494,494]
    dataFiles = [os.path.join('data','dmc2021n{:06d}.hdf'.format(no)) for no in fileNumbers]

    ds = DataSet.DataSet(dataFiles)
    
    assert(len(ds) == len(dataFiles))
    assert(ds[0].fileName == os.path.split(dataFiles[0])[-1])


    # load single file and check that it is equal to the corresponding file in ds
    ds2 = DataSet.DataSet(dataFiles[-1])
    assert(len(ds2) == 1)
    assert(ds2[0].fileName == os.path.split(dataFiles[-1])[-1])
    assert(ds2[0] == ds[-1])

    # Find length before appending
    length = len(ds)
    ds.append(dataFiles)

    # Length is supposed to be both
    assert(len(ds)==length+len(dataFiles))

    # Also works for adding data files directly (both in list and as object)
    df = ds2[0]
    ds.append([df])
    assert(len(ds)==length+len(dataFiles)+1)
    ds.append(df)
    assert(len(ds)==length+len(dataFiles)+2)

    # Deletion
    del ds[-1]
    assert(len(ds)==length+len(dataFiles)+1)

def test_plot():

    fileNumbers = [565]
    dataFiles = [os.path.join('data','dmc2021n{:06d}.hdf'.format(no)) for no in fileNumbers]


    ds = DataSet.DataSet(dataFiles)
    ds[0].monitor[0] = np.array([1.0])
    
    Ax, bins, intensity, error, monitor = ds.plotTwoTheta()

    Ax,*_ = ds.plotTwoTheta(correctedTwoTheta=False)

    # Calculate bins, intensity, error without plotting

    bins2, intensity2, error2, monitor2 = ds.sumDetector()

    print(np.sum(intensity-intensity2))

    assert(np.all(np.isclose(bins,bins2)))
    assert(np.all(np.isclose(intensity,intensity2,equal_nan=True)))
    assert(np.all(np.isclose(error,error2,equal_nan=True)))
    assert(np.all(np.isclose(monitor,monitor2)))
    

def test_2d():
    fileNumbers = [494,494]
    dataFiles = [os.path.join('data','dmc2021n{:06d}.hdf'.format(no)) for no in fileNumbers]

    ds = DataSet.DataSet(dataFiles=dataFiles)

    files = len(fileNumbers)
    assert(ds[0].counts.shape == (1,128,1152))

    ax1 = ds.plotTwoTheta(correctedTwoTheta=False)
    ax2 = ds.plotTwoTheta(correctedTwoTheta=True)


def test_kwargs():
    
    fileNumbers = [494,494]
    dataFiles = [os.path.join('data','dmc2021n{:06d}.hdf'.format(no)) for no in fileNumbers]

    ds = DataSet.DataSet(dataFiles=dataFiles)

    try:
        _ = ds.sumDetector(corrected=False)
        assert(False)
    except AttributeError as e:
        #assert(e.args[0] == 'Key-word argument "corrected" not understood. Did you mean "correctedTwoTheta"?')
        assert(True)

    ds = DataSet.DataSet(dataFiles=dataFiles)

    try:
        _ = ds.plotTwoTheta(corrected=False,fmt='.-')
        assert(False)
    except AttributeError as e:
        #assert(e.args[0] == 'Key-word argument "corrected" not understood. Did you mean "correctedTwoTheta"?')
        assert(True)



def test_export_PSI_format(folder='data'):  
    
    if folder is None:
        folder = os.getcwd()
        
    fileNumbers = [565]
    
    dataFiles = [os.path.join(folder,'dmc2021n{:06d}.hdf'.format(no)) for no in fileNumbers]
    print(dataFiles)
    ds = DataSet.DataSet(dataFiles)
    
    for df in ds:
        if np.any(np.isnan(df.monitor)) or np.any(np.isclose(df.monitor,0.0)):
            df.monitor = np.ones_like(df.monitor)
    
    ds.export_PSI_format(outFile="testfile")
    
    assert(os.path.exists("testfile.dat") == True and os.stat("testfile.dat").st_size != 0)
    
    os.remove("testfile.dat")

#test_export_PSI_format()    


def test_export_xye_format(folder='data'):
    
    if folder is None:
        folder = os.getcwd()
        
    fileNumbers = [565]
    
    dataFiles = [os.path.join(folder,'dmc2021n{:06d}.hdf'.format(no)) for no in fileNumbers]
    
    ds = DataSet.DataSet(dataFiles)
    
    for df in ds:
        if np.any(np.isnan(df.monitor)) or np.any(np.isclose(df.monitor,0.0)):
            df.monitor = np.ones_like(df.monitor)
    
    ds.export_xye_format(outFile="testfile")
    
    assert(os.path.exists("testfile.xye") == True and os.stat("testfile.xye").st_size != 0)
    
    os.remove("testfile.xye")

#test_export_xye_format() 



def test_add():
    
    DataSet.add(565,566,outFile='test_add',folder='data',dataYear=2021)
    
    assert(os.path.exists("test_add.dat") == True and os.stat("test_add.dat").st_size != 0)
    assert(os.path.exists("test_add.xye") == True and os.stat("test_add.xye").st_size != 0)
    os.remove("test_add.dat")
    os.remove("test_add.xye")

# test_add()    

def test_export():
    
    DataSet.export(565,outFile='test_export', xye=True,folder='data',dataYear=2021)
    
    assert(os.path.exists("test_export.dat") == True and os.stat("test_export.dat").st_size != 0)
    assert(os.path.exists("test_export.xye") == True and os.stat("test_export.xye").st_size != 0)
    os.remove("test_export.dat")
    os.remove("test_export.xye")

# test_export()


def test_export_from(folder = 'data'):
    """
    Runs the last two files in the data folder. These files must be powder scans for test to work    
    """

    if folder is None:
        folder = os.getcwd()
        
    hdf_files = [f for f in os.listdir(folder) if f.endswith('.hdf')]
    last_hdf = hdf_files[-1]
    numberOfFiles = int(last_hdf.strip('.hdf').split('n')[-1])
    file1 = f"DMC_{numberOfFiles-2}"
    file2 = f"DMC_{numberOfFiles-1}"
 
    DataSet.export_from(numberOfFiles-2,sampleName=False, xye=True,temperature=False,folder='data',dataYear=2022)
    
    assert(os.path.exists(file1+'.dat') == True and os.stat(file1+'.dat').st_size != 0)
    assert(os.path.exists(file2+'.dat') == True and os.stat(file2+'.dat').st_size != 0)   
    assert(os.path.exists(file1+'.xye') == True and os.stat(file1+'.xye').st_size != 0)
    assert(os.path.exists(file2+'.xye') == True and os.stat(file2+'.xye').st_size != 0)       
    os.remove(file1+'.dat')
    os.remove(file2+'.dat')
    os.remove(file1+'.xye')
    os.remove(file2+'.xye')


def test_export_from_to():
    
    DataSet.export_from_to(565,566,sampleName=False, xye=True,temperature=False,folder='data',dataYear=2021)
    
    assert(os.path.exists("DMC_565.dat") == True and os.stat("DMC_565.dat").st_size != 0)
    assert(os.path.exists("DMC_565.xye") == True and os.stat("DMC_565.xye").st_size != 0)
    assert(os.path.exists("DMC_566.dat") == True and os.stat("DMC_566.dat").st_size != 0)
    assert(os.path.exists("DMC_566.xye") == True and os.stat("DMC_566.xye").st_size != 0)
    os.remove("DMC_565.dat")
    os.remove("DMC_565.xye")
    os.remove("DMC_566.dat")
    os.remove("DMC_566.xye")

# test_export_from_to()


def test_export_list():
    
    DataSet.export([565],outFile='test_export_list', xye=True,folder='data',dataYear=2021)
    
    assert(os.path.exists("test_export_list.dat") == True and os.stat("test_export_list.dat").st_size != 0)
    assert(os.path.exists("test_export_list.xye") == True and os.stat("test_export_list.xye").st_size != 0)
    os.remove("test_export_list.dat")
    os.remove("test_export_list.xye")
    
# test_export_list()

def test_subtract_PSI():
    
    DataSet.subtract_PSI('DMC_565','DMC_566',outFile='test_subtract_PSI',folder='data')
    
    assert(os.path.exists("test_subtract_PSI.dat") == True and os.stat("test_subtract_PSI.dat").st_size != 0)
    os.remove("test_subtract_PSI.dat")

def test_subtract_xye():
    
    DataSet.subtract_xye('DMC_565','DMC_566',outFile='test_subtract_xye',folder='data')
    
    assert(os.path.exists("test_subtract_xye.xye") == True and os.stat("test_subtract_xye.xye").st_size != 0)
    os.remove("test_subtract_xye.xye")


def test_subtract():
    
    DataSet.subtract('DMC_565','DMC_566',outFile='test_subtract',folder='data')

    assert(os.path.exists("test_subtract.dat") == True and os.stat("test_subtract.dat").st_size != 0)
    assert(os.path.exists("test_subtract.xye") == True and os.stat("test_subtract.xye").st_size != 0)
    os.remove("test_subtract.dat")
    os.remove("test_subtract.xye")

def test_updateDataFileParameters():


    fileNumbers = [565,565,566]
    dataFiles = [os.path.join('data','dmc2021n{:06d}.hdf'.format(no)) for no in fileNumbers]

    ds = DataSet.DataSet(dataFiles)
    oldWavelength = [df.wavelength for df in ds]

    newWavelength = 10
    ds.updateDataFiles('wavelength',newWavelength)
    assert(np.all([np.isclose(newWavelength,df.wavelength) for df in ds]))

    try:
        ds.updateDataFiles('WaveLength',newWavelength) # Doesn't have the attribute
        assert False
    except AttributeError:
        assert True

    try:
        ds.updateDataFiles('wavelength',[10,10]) # Wrong length of parameters
        assert False
    except AttributeError:
        assert True


    newValues = np.random.rand(len(ds))
    ds.updateDataFiles('twoThetaPosition',newValues)
    assert(np.all([np.isclose(nV,df.twoThetaPosition) for nV,df in zip(newValues,ds)]))
