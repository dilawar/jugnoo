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

def edges( filename, **kwargs ):
    print('[INFO] Detecting edge in %s' % filename)
    img = cv2.imread(filename, 0)
    high = kwargs.get('threshold_high', 200)
    low = kwargs.get('threshold_low', high / 2 )
    edges = cv2.Canny( img, low, high, L2gradient = True )
    if kwargs.get('debug', False):
        img = np.concatenate( (img, edges) )
        outfile = '%s_out.png' % filename
        cv2.imwrite('%s_out.png' % filename)
        print('[DEBUG] Wrote edges and original file to %s ' % outfile )

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''description'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--file', '-f'
        , required = True
        , help = 'Image file'
        )
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    edges( args.file, **vars(args) )
