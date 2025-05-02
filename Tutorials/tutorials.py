#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:03:12 2021

@author: Jakob Lass

Tool to replace batch in order to fully port development to windows
"""

import os,sys,shutil

DIR = str(os.path.dirname(__file__))

try:
    os.remove(os.path.join('Tutorials','tutorialList.txt'))
except FileNotFoundError:
    pass

files = [f.path for f in os.scandir(DIR) if f.is_file()]

for f in files:
    if not f[-2:] == 'py':
        continue
    if 'tutorials.py' in f:
        continue
    os.system('python '+f)

    
    

for folder in ['.cache','.pyteste_cache','__pycache']:
    try:
        shutil.rmtree(os.path.join(DIR,folder))
    except FileNotFoundError:
        pass


# Generate correct tutorials.rst file
tutorialFile = os.path.join('docs','Tutorials','Tutorials.rst')

mainText = """.. _Tutorials:

Tutorials
---------
Below is a list of tutorials highlighting the capabilities of DMCpy in regards to a variety of different applications. It ranges from an overall explanation 
of how to generate *DataFile* and *DataSet* objects over direct plotting of the detector images to generation of the UB matrix, cuts and background subtraction. 
Below are all of the tutorials for DMCpy. Some sample data has been made available `here`_

.. toctree::
   :maxdepth: 1
"""

with open(os.path.join('Tutorials','tutorialList.txt'),'r') as f:
    tutorials = f.readlines()
            
tutorialText = []
for tut in tutorials:
    common = os.path.commonpath([tut,os.path.abspath(tutorialFile)]).strip()
    tutorialText.append(str(os.path.relpath(tut,common)).replace('\\','/'))


text = mainText+'\n   '+'\n   '.join(tutorialText)+'\n\n'+'.. _here: https://www.dropbox.com/scl/fo/86czexjmg4r7wvqbvqimo/AIqE7nmLQSoa4EGA8FX0Pl0?rlkey=p9d0i9cfcc1pzdoz3zutsjf1v&st=ls3avd3i&dl=0\n\n'



with open(tutorialFile,'w') as f:
    f.write(text)

