#!/usr/bin/env python3

__author__ = "nicoroble"
__version__ = "0.1.0"
__license__ = "MIT"

import os

def main():
    """ Main entry point of the app """
    models = ['basefinal', 'final01', 'augmented02', 'multitaskfinal']

    for model in models:
        command = F'python measure-accuracy.py -m {model}'
        os.system(command)

if __name__ == "__main__":
    """ This is executed when run from the command line """
    main()