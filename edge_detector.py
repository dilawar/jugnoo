#!/usr/bin/env python

"""edge_detector.py: 

Detect edges in an image.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import cv2 
import numpy as np
import pylab
import helper 

debug_level_ = 0
def debug__( msg, level = 1 ):
    # Print debug message depending on level
    global debug_level_ 
    if type(msg) == list: msg = '\n|- '.join(msg)
    if level >= debug_level_: print( msg )

def edges( filename, **kwargs ):
    global debug_level_
    debug_level_ = kwargs.get( 'debug', 0)
    debug__('[INFO] Detecting edges in %s' % filename)
    img = cv2.imread(filename, 0)
    high = kwargs.get('threshold_high', 200)
    low = kwargs.get('threshold_low', high / 2 )
    edges = cv2.Canny( img, low, high, L2gradient = True )
    if kwargs['debug'] > 0:
        outfile = kwargs.get('outfile', None) or '%s_edges.png' % filename
        helper.plot_images( { 'original' : img, 'edges' : edges } )

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Detect edges in an image'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', '-i'
        , required = True
        , help = 'Input file'
        )
    parser.add_argument('--output', '-o'
        , required = False
        , help = 'Output file'
        )
    parser.add_argument( '--debug', '-d'
        , required = False
        , default = 0
        , type = int
        , help = 'Enable debug mode. Default 0, debug level'
        )
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    edges( args.input,  **vars(args) )

