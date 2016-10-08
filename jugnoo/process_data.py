#!/usr/bin/env python3

"""process_data.py: 

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2016, Dilawar Singh"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import os
import matplotlib.pyplot as plt
import igraph as ig

def main():
    """docstring for main"""
    g = ig.load( sys.argv[1] )
    print( g)

if __name__ == '__main__':
    main()
