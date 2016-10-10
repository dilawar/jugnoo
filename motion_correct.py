#!/usr/bin/env python

"""motion_correct.py:

Correct motion.

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
import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import cv2


def main( datafile ):
    stack = np.load( datafile ).T
    f0 = cv2.medianBlur( stack[0], 11 )
    # f0 = stack[1]
    h, w = 50, 50
    for f in stack[:3]:
        f = cv2.medianBlur( f, 11 )
        print np.sum( f - f0 )
        ff = f[h:f.shape[0]-h,w:f.shape[1]-w]
        conv = scipy.signal.fftconvolve( f0, ff / ff.max(), 'valid' )
        y, x = cv2.minMaxLoc( conv )[3]
        print x, y 


if __name__ == '__main__':
    datafile = sys.argv[1]
    main( datafile )
