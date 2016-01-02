#!/usr/bin/env python

"""contour_detection.py: 

Detect contours in an image

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

def contours( filename, **kwargs ):
    print('[INFO] Detecting contours in %s' % filename)
    img = cv2.imread(filename, 0)
    high = kwargs.get('threshold_high', img.max() )
    low = kwargs.get('threshold_low', img.mean() +  img.std() )
    ret, thres = cv2.threshold( img, low, high, 0)
    cnts, heir = cv2.findContours(thres, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if kwargs.get('debug', False):
        cntImg = np.zeros( img.shape )
        print cnts
        cv2.drawContours( cntImg, cnts, -1, 255)
        cv2.imshow( 'cnt', cntImg )
        cv2.waitKey( 0 )
        # img = np.concatenate( (img, cntImg) )
        outfile = '%s_contours.png' % filename
        cv2.imwrite( outfile, thres )
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
    parser.add_argument('--debug', '-d'
        , required = False
        , default = True
        , action = 'store_true'
        , help = 'Help'
        )
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    contours( args.file, **vars(args) )
