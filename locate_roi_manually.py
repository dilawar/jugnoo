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

rois = []

# Globals for drawing circles.
points = []

def draw_roi(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        img_[y, x] =  255
        points.append([x,y])

    elif event == cv2.EVENT_LBUTTONDBLCLK:
        # get rid of first point which is from last double click.
        pts = np.array( points[1:] )
        points = []
        circle = cv2.minEnclosingCircle( pts )
        (row, col), r = circle
        rois.append([col,row,r])
        cv2.circle(img_, (int(row), int(col)), int(r), 255, 1)

# Create a black image, a window and bind the function to window
img_ = None
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_roi)

def write_roi( ):
    global circles
    roiFile = kwargs.get('output') or '%s_manual_rois.csv' % filename
    txt = [ 'col,row,radius' ]
    for c in circles:
        txt.append( '%d,%d,%d' % tuple(c))
    with open(roiFile, 'w') as f:
        f.write( '\n'.join( txt ) )
    print('[INFO] Wrote all ROIs to %s' % outfile )

def main( filename, **kwargs):
    global img_
    frames = fr.read_frames( filename )
    img_ = np.zeros( frames[0].shape )
    if kwargs.get('stride') == -1:
        for f in frames:
            img_ += f

    img_ = fr.to_grayscale( img_ )
    while(1):
        cv2.imshow('image', img_)
        k = cv2.waitKey(1) & 0xFF
        if ord('q') == k:
            break
    cv2.destroyAllWindows()
    write_roi( )

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Locate ROI's manually'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', '-i'
        , required = True
        , help = 'Input file'
        )
    parser.add_argument('--output', '-o'
        , required = False
        , help = 'Output result file (csv)'
        )
    parser.add_argument( '--debug', '-d'
        , required = False
        , default = 0
        , type = int
        , help = 'Enable debug mode. Default 0, debug level'
        )
    parser.add_argument('--stride', '-s'
        , required = False
        , default = -1
        , help = 'Help'
        )
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    main( args.input, ** vars( args ))
