#!/usr/bin/env python

"""locate_roi_manually.py: 

Locate roi manually.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

from image_analysis import frame_reader as fr
import cv2
import numpy as np

def main( filename, **kwargs):
    global img_
    frames = fr.read_frames( filename )
    img_ = np.zeros( frames[0].shape )
    circles = []
    with open(kwargs['roi_file'], 'r') as f:
        lines = f.read().split('\n')
        for l in lines:
            circles.append( [ int(x) for x in l.split(',') ] )
    print circles

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Locate ROI's manually'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', '-i'
        , required = True
        , help = 'Input file (tiff/avi)'
        )
    parser.add_argument('--roi_file', '-r'
        , required = False
        , help = 'ROI file (csv)'
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
    main( args.input, ** vars( args ))
