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
import pylab

def get_baseline( vec ):
    # baseline = mean of first 5 frames.
    return vec - vec[0:5].mean()

def get_rois( roifile ):
    print('[INFO] Reading rois from %s' % roifile)
    rois = np.genfromtxt(roifile, delimiter=',', comments='#', skip_header=True)
    rois_ = []
    for r in rois:
        rois_.append( list(r) )
    rois = sorted(rois_, key = lambda x: x[1])
    return rois

def compute_df_by_f( roi, frames ):
    """Compute df/f for given ROI in each frame and return the array """
    col, row, r = roi
    p1, p2 = (row-r, col-r), (row+r, col+r) 
    mask = np.zeros( shape = frames[0].shape )
    # Holder to create image with given df/f ratio
    img = np.zeros( shape = mask.shape )
    dfbyf = np.zeros( len(frames) )
    for i, f in enumerate(frames):
        # Draw a filled circle on mask
        cv2.circle( mask, (int(row), int(col)), int(r), 1, -1) # Filled circle -1
        img[ mask == 1 ] = f[ mask == 1 ]
        #cv2.imshow( 'mask', img )
        dfbyf[i] =  img.mean() 
    #cv2.waitKey( 0 )
    baseline = get_baseline( dfbyf )
    return baseline 

def main( imagefile, roi_file, outfile = None):
    global img_
    rois = get_rois( roi_file )
    frames = fr.read_frames( imagefile )
    img_ = np.zeros( frames[0].shape )

    # This image keep one row for each df/f, and one column for each frame 
    dfbyfImg = np.zeros( shape = ( len(rois), len(frames) ) )
    for i, roi in enumerate(rois):
        vec = np.array(compute_df_by_f( roi, frames ))
        dfbyfImg[i,:] = vec

    outfile = '%s_dfbyf.dat' % ( outfile or imagefile )
    np.savetxt( outfile, dfbyfImg, delimiter=',' )
    print('[INFO] Writing dfbyf data to %s' % outfile)
    cx = pylab.imshow( dfbyfImg, cmap = pylab.cm.hot, aspect = 'auto' )
    pylab.colorbar( cx , orientation = 'horizontal' )

    pylab.title = 'df/f in ROIs'
    pylab.xlabel( '# frame ')
    pylab.ylabel( '# roi ')
    outfile = '%s_df_by_f.png' % ( outfile or imagefile )
    pylab.savefig( outfile )
    print('[INFO] Done saving datafile to %s ' % outfile)


if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Locate ROI's manually'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--input', '-i'
        , required = True
        , help = 'Input file (tiff/avi)'
        )
    parser.add_argument('--roifile', '-r'
        , required = True
        , help = 'ROI file (csv)'
        )
    parser.add_argument( '--debug', '-d'
        , required = False
        , default = 0
        , type = int
        , help = 'Enable debug mode. Default 0, debug level'
        )
    parser.add_argument('--outfile', '-o'
        , required = False
        , help = 'Output file path'
        )

    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    main( args.input, args.roifile, args.outfile)
