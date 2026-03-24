import argparse
import glob
import os
import numpy as np

OpossibleCommands = ['Nope','None','I don\'t know']

list_of_commands = glob.glob(os.path.join('DMCpy','CommandLineScripts','*.py')) # * means all if need specific format then *.csv

def oxfordlist(ls):
    if len(ls) == 1:
        return ls[0]
    if len(ls) == 2:
        return ' and '.join(ls)
    return ', '.join(ls[-1])+'and '+ls[-1] 

## python files to skip
skip = ['DMCpy.py','__init__.py']

list_of_valid_commands = [command for command in list_of_commands if not os.path.split(command)[-1] in skip]
list_of_valid_command_names = [os.path.splitext(os.path.split(command)[-1])[0] for command in list_of_valid_commands]

parser = argparse.ArgumentParser(description="Collection of DMCpy tools for the command line")
parser.add_argument("task", nargs='?', default='help', type=str, help="Type of task to be performed. Possible tasks are: {}. Run without argument to see help menu.".format(oxfordlist(list_of_valid_command_names)))
parser.add_argument('additional', nargs=argparse.REMAINDER)

args = parser.parse_args()

try:
    idx = list_of_valid_command_names.index(args.task)

    arguments = dict(vars(args))
    del arguments['task']

    print('wanting to call {} with arguments {}'.format(list_of_valid_commands[idx],arguments))
    os.system('python {} {}'.format(list_of_valid_commands[idx],args.additional))
except:
    print('Command not understood from DMCpy')
    print(args)
