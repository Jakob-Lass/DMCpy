import argparse

OpossibleArguments = ['Nope','None','I don\'t know']

parser = argparse.ArgumentParser(description="Collection of DMCpy tools for the command line")
#parser.add_argument("task", nargs='?', default='help', type=str, help="Type of task to be performed. Possible tasks are: {}. Run without argument to see help menu.".format(OpossibleArguments))


args = parser.parse_args()

print(args)
