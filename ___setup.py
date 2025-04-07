from setuptools import setup
import setuptools
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))
operatingSystem = sys.platform


with open(os.path.join(_here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


installFolder = os.path.abspath(os.path.join(os.path.split(setuptools.__file__)[0],'..'))
pythonPath =  os.path.relpath(os.path.join(installFolder,'DMCpy'),sys.base_prefix)

setup(
    name='DMCpy',
    version='1.0.1',
    description=('Python software packaged designed for reduction of neutron powder and single crystal diffraction data from DMC at PSI.'),
    long_description=long_description,
    author='Jakob Lass, Sam Moody, Øystein S. Fjellvåg',
    author_email='jakob.lass@psi.ch',
    url='https://www.psi.ch/en/sinq/dmc/',
    license='MPL-2.0',
    data_files = [(pythonPath, ["LICENSE.txt"])],
    #packages=['DMCpy','src/CommandLineScripts'],
    package_data={"src": ["*.dat"]},
    entry_points = {
        "console_scripts": ['DMCSpy=DMCpy.CommandLineScripts.DMCS:main'],
        },
    python_requires='>=3.6',
    install_requires=['matplotlib','numpy>=1.14','h5py>=2.5','scipy','datetime','pandas','future',
                    'pip>=20'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'],
    )
