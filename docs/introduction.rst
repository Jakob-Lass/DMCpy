
Introduction to the DMC Instrument
==================================

DMC is a cold neutron diffractometer specialized in magnetic powder and single crystal diffraction. It is located in SINQ in the guide hall at the Paul Scherrer Institute (`DMC Web Page <https://www.psi.ch/en/sinq/dmc>`_)

With the upgrade to of the instrument currently ongoing, a new 2D detector is installed. It has an in-plane coverage of 130\ :sup:`o` and a vertical height of 20 cm. It is located 
80 cm from the sample position giving an out-of-plane coverage of +- 7.1\ :sup:`o`.



DMCpy
^^^^^

For the instrument, a python package **DMCpy** has been written. It is written in pure python and supports the versions indicated on the main page. Some dependencies on other libraries exists, including: MatplotLib, Scipy, h5py, and Pandas.
These are viewed as major packages where support and an ongoing development is expected. 

The package can be installed using the Python Package manager `PyPI <https://pypi.org/>`_. The recommended way as follows:

1. Install Conda or miniconda as explained on the (`Conda Web Page <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html/>`_) with the downloads found (`Anaconda Download <https://www.anaconda.com/products/individual>`_) or according to your institution
2. In either the Prompt or through the Navigator create a custom environment

   * In the prompt write :code:`conda create -n DMCpy python=3.9` where the name of the environment is chosen to be *DMCpy* and python version 3.9. See documentation main page for currently supported python versions
   * Or in the navigator, click the *Environments* tab and then *create*. In the *Create new environment* choose a name, e.g. *DMCpy*, and a python version, e.g. 3.9.

3. Activate the newly created environment

   * In the prompt write :code:`conda activate DMCpy`. Notice that the *(base)* text changes to *(DMCpy)*
   * Or through the navigator, click the name of the environment

4. Install the DMCpy package 

   * In the prompt write :code:`pip install DMCpy` and PyPI takes care of the installation of the DMCpy dependencies, pathing, and setup of all the packages.
   * Or through the navigator by starting a prompt clicking the *Open terminal* button and following the above described procedure

5. If a jupyter notebook is wanted continue in the prompt to install it by 'pip install jupyter' or by going to the *Home* tab in the navigator an through it install jupyter notebook.

6. If you prefer to work in spyder, you must install it in the new environment by: 'pip install spyder' 

After the initial installation, try running one of the :ref:`Tutorials` to check that the installation was successful. Make sure to activate the created environment before running tutorials or notebooks.

To update to the newest version of DMCpy, you can run this command:  :code:`pip install --upgrade DMCpy`


IPython
^^^^^^^

The DMCpy software package makes use of many features of the interactive part of matplotlib. Thus, if the code/tutorials are run through an IPython kernel, these might be absent. However, including the following code snippet to the top of the scripts changes the IPython matplotlib back-end to be interactive:

.. code-block:: python

    try:
        import IPython
        shell = IPython.get_ipython()
        shell.enable_matplotlib(gui='qt')
    except:
        pass

This block can in principal also be included for regular python kernels as it will then through an exception and pass. If the 'qt' back-end is not available, one could try a number of others; For a list run "matplotlib.rcsetup.interactive_bk" after import of matplotlib. 