#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 10:31:10 2019

@author: lass
"""

import sys
import glob
import os

if len(sys.argv)==1:
	
	list_of_files = glob.glob('dist/*') # * means all if need specific format then *.csv
	try:
		latest_file = os.path.split(max(list_of_files, key=os.path.getctime))[-1]
	except ValueError:
		raise ValueError('It seems that there are no wheels in this project (no folder called "dist"). Please run "make wheel" first')


	latestVersion = [int(x) for x in latest_file.split('-')[-1].replace('.tar.gz','').split('.')]

	latestVersion[-1]+=1
	version = '{}.{}.{}'.format(*latestVersion)
else:
	version = sys.argv[1]

print('Updating to version ',version)

# update to new version in setup.py
with open('pyproject.toml') as f:
	lines = f.readlines()
	writeLines = ''
	for l in lines:
		if l.find('version = "')!=-1:
			l ='version = "'+version+'"\n'
		writeLines+=l
			
with open('pyproject.toml','w') as f:
	f.write(writeLines)

with open('src/__init__.py') as f:
	lines = f.readlines()
	writeLines = ''
	for l in lines:
		if l.find("__version__ = ")!=-1:
			idx = l.find('__version__ = ')
			l = l[:idx] + "__version__ = '"+version+"'\n"
		writeLines+=l
        
with open('src/__init__.py','w') as f:
	f.write(writeLines)

with open('docs/conf.py') as f:
	lines = f.readlines()
	writeLines = ''
	for l in lines:
		if l.find("version = u'")!=-1:
			l = "version = u'"+version+"'\n"
		elif l.find("release = u'")!=-1:
			l = "release = u'"+version+"'\n"
		writeLines+=l
			
with open('docs/conf.py','w') as f:
	f.write(writeLines)

with open('test/init.py') as f:
	lines = f.readlines()
	writeLines = ''
	for l in lines:
		if l.find("__version__")!=-1:
			idx = l.find('__version__==')
			l = l[:idx] + "__version__=='"+version+"')\n"
		writeLines+=l
        
with open('test/init.py','w') as f:
	f.write(writeLines)
if False:
    os.system("git add DMCpy/__init__.py test/init.py setup.py")
    os.system('git commit -m "Update to version {}"'.format(version))
    os.system('git tag -a {} -m "{}"'.format(version,version))
