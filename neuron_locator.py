#!/usr/bin/env python

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import config 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy.stats as stat
import os
import sys
import glob
import datetime
import cv2
import helper
from image_analysis import edge_detector 
from image_analysis import contour_detector 
import environment as e
import roi
import image_reader

import logging
logger = logging.getLogger('')

def init( ):
    e.save_direc_ = os.path.join( '.', '_results_%s' %
            os.path.split(e.args_.input)[1])
    if os.path.isdir( e.save_direc_ ):
        return
        for f in glob.glob( '%s/*' % e.save_direc_ ):
            os.remove(f)
    else:
        logging.info("Creating dir %s for saving data" % e.save_direc_)
        os.makedirs( e.save_direc_ )

def to_grayscale( img ):
    if img.max() >= 256.0:
        logging.debug("Converting image to grayscale")
        logging.debug("Max=%s, min=%s, std=%s"% (img.max(), img.min(),
            img.std()))
        img = 255 * ( img / float( img.max() ))
    gimg = np.array(img, dtype=np.uint8)
    return gimg

def get_edges( frame ):
    cannyFrame = to_grayscale( frame )
    edges = cv2.Canny( cannyFrame
            , config.elow, config.ehigh
            , L2gradient = True
            , apertureSize = config.canny_window_size
            )
    return edges

def get_activity_vector( frames ):
    """Given a list of frames, return the indices of frames where there is an
    activity
    """
    meanActivity = [ x.mean() for x in frames ]
    # Now get the indices where peak value occurs. 
    activity = sig.argrelextrema( np.array(meanActivity), np.greater)[0]
    # activity = activity[::3]
    actFlie = os.path.join( e.save_direc_, 'activity_peak.csv')
    logging.info("Writing activity to %s" % actFlie)
    header = "frame index"
    np.savetxt(actFlie, activity, delimiter=',', header=header)
    logging.info("... Wrote!")
    return activity

def threshold_frame( frame, nstd = None):
    # Change the parameter to one's liking. Currently low threshold value is 3
    # std more than mean.
    mean = int(frame.mean())
    std = int(frame.std())
    if nstd is None:
        nstd = 3
    low = max(0, mean + (nstd * std))
    high = int( frame.max() )
    logging.debug("Thresholding at %s + %s * %s" % (mean, nstd, std))
    logging.debug("|-  low, high = %s, %s" % (low, high))
    frame = stat.threshold( frame, low, high, newval = 0)
    return to_grayscale( frame )

def save_image( filename, img, **kwargs):
    """Store a given image to filename """
    if e.args_.debug:
        img = to_grayscale( img )
        outfile = os.path.join( e.save_direc_, filename )
        logging.info( 'Saving image to %s ' % outfile )
        cv2.imwrite( outfile , img )
        return img

def write_ellipses( ellipses ):
    outfile = os.path.join( e.save_direc_, 'bounding_ellipses.csv' )
    with open( outfile, 'w' ) as f:
        f.write("x,y,major,minor,rotation\n")
        for e in ellipses:
            (y, x), (minor, major), angle = e
            f.write('%s,%s,%s,%s,%s\n' % (x, y, major, minor, angle))
    logging.info("Done writing ellipses to %s" % outfile )

def get_rois( frames, window):
    # Compute the region of interest.
    e.shape_ = frames[0].shape

    activityVec = get_activity_vector( frames )
    e.images_['activity'] = np.array( activityVec )

    # activityVec contains the indices where we see a local maxima of mean
    # activity e.g. most likely here we have a activity at its peak. Now we
    # collect few frames before and after it and do the rest.
    # logger.debug("Activity vector: %s" % activityVec )
    allEdges = np.zeros( e.shape_ )
    roi = np.zeros( e.shape_ ) 
    for i in activityVec:
        low = max(0, i-window)
        high = min( e.shape_[0], i+window)
        bundle = frames[low:high]
        sumAll = np.zeros( e.shape_ )
        for f in bundle:
            ff = threshold_frame( f, nstd = 2)
            sumAll += ff
        edges = get_edges( sumAll )
        merge_image = np.concatenate( (to_grayscale(sumAll), edges), axis=0)
        save_image( 'edges_%s.png' % i, merge_image)
        #  Also creates a list of acceptable cells in each frame.
        cellImg = compute_cells( edges )
        save_image( 'cell_%s.png' % i, cellImg )
        roi += cellImg
        allEdges += edges 

    e.images_['all_edges'] = allEdges
    e.images_['rois'] = to_grayscale(roi)

    save_image( 'all_edges.png', allEdges, title = 'All edges')
    save_image( 'rois.png', roi )

    #  Use this to locate the clusters of cell in all frames. 
    cnts, cntImgs = find_contours( to_grayscale(roi), draw = True, fill = True)
    e.images_['bound_area'] = get_edges( cntImgs )
    logger.info("Done processing all the contours. Computed the bounded areas")

def find_contours( img, **kwargs ):
    logger.debug("find_contours with option: %s" % kwargs)
    # Just return external points of contours, and apply Ten-Chin chain
    # approximation algorithm. 
    
    contours, h = cv2.findContours(img
            , cv2.RETR_LIST              # No Homo Hierarichus!
            , cv2.CHAIN_APPROX_TC89_KCOS # Apply Tin-Chen algo to return
                                         # dominant point of curve.
            )
    if kwargs.get('hull', True):
        logger.debug("Approximating contours with hull")
        contours = [ cv2.convexHull( x ) for x in contours ]

    if kwargs.get('filter', 0):
        contours = filter(lambda x:len(x) >= kwargs['filter'], contours)

    contourImg = None
    if kwargs.get('draw', False):
        contourImg = np.zeros( img.shape, dtype=np.uint8 )
        cv2.drawContours( contourImg, contours, -1, 255, 1)

    if kwargs.get('fill', False):
        for c in contours:
            cv2.fillConvexPoly( contourImg, c, 255 )
    return contours, contourImg

def acceptable( contour ):
    """Various conditions under which a contour is not a cell """
    # First fit it with an ellipse
    roi = roi.ROI( contour )
    # If area of contour is too low, reject it.
    diam = 2.0 * roi.radius
    if diam < config.min_roi_diameter or diam > config.max_roi_diameter:
        logger.debug( "Rejected ROI. Diameter is low or high (%s)" % diam)
        continue

    # If the lower axis is 0.7 or more times of major axis, then aceept it.
    if roi.eccentricity < 0.7:
        msg = "Contour %s is rejected because " % contour
        msg += "axis ration (%s) of cell is too skewed" % roi.eccentricity
        logger.debug( msg )
        return False

    e.cells_.append( roi )
    return True

def compute_cells( image ):
    thresholdImg = threshold_frame( image, nstd = 3 )
    contourThres = config.min_points_in_contours
    contours, contourImg = find_contours(thresholdImg
            , draw = True
            , filter = contourThres
            , hull = True
            )

    img = np.zeros( contourImg.shape, dtype = np.uint8 )
    for c in contours:
        if acceptable( c ):
            cv2.fillConvexPoly( img, c, 255)

    # Now fetch the contours from this image. 
    contours, contourImg = find_contours( img, draw = True, hull = True )
    return contourImg

def df_by_f( roi, frames, roi_index = None ):
    msg = 'ROI %s' % str(roi)
    if roi_index is not None:
        msg = ('%s : ' % roi_index) + msg
    logger.info( msg )

    yvec = []
    for f in frames:
        col, row, w, h = roi
        area = f[row:row+h,col:col+w]
        yvec.append( area.mean() )

    yvec = np.array(yvec, dtype=np.float)
    df = yvec - yvec.min()
    return np.divide(100 * df, yvec) 

def df_by_f_data( rois, frames ):

    dfmat = np.zeros( shape = ( len(rois), len(frames) ))
    for i, r in enumerate(rois):
        vec = df_by_f( r, frames, i )
        dfmat[i,:] = vec
    
    outfile = '%s/df_by_f.dat' % e.save_direc_
    comment = 'Each column represents a ROI'
    comment += "\ni'th row is the values of ROIs in image senquence i"
    np.savetxt(outfile, dfmat.T, delimiter=',', header = comment)
    save_image( 'df_by_f.png', dfmat)
    logger.info('Wrote df/f data to %s' % outfile)
    return dfmat

def merge_or_reject_cells( cells ):
    restult = []
    coolcells, areas = [], []

    # sort according to area
    cells = sorted( cells, key = lambda x: x.area )

    # Now remove all those cells which are inside another cells. They should be
    # close enough in size.
    logger.info("Removing overlapping cells")
    cells = helper.remove_contained_cells( cells )
    logger.info("== Done removing overlapping cells")

    logger.info("Removing duplicate ROIs")
    cells = helper.remove_duplicates( cells )
    logger.info("== Done removing duplicated")
    return cells

def get_roi_containing_minimum_cells( ):
    neuronImg = np.zeros( shape = e.shape_ )
    coolcells = []

    # Now we need reject some of these rectangles.
    logger.info("Merging or rejecting ROIs to find suitable cells")
    coolerCells = merge_or_reject_cells( coolcells )
    logger.info("Done merging or rejecting ROIs")

    # create an image of these rois.
    for c in coolerCells:
        center, radius = c.center, int(c.radius)
        cv2.circle( neuronImg, ( int(center[0]), int(center[1])), radius, 255, 1)
    e.images_['neurons'] = neuronImg

    return coolerCells 

def process_input( ):
    inputfile = e.args_.input

    logger.info("Processing %s" % inputfile)
    frames = image_reader.read_frames( inputfile )
    
    # get the summary of all activity
    summary = np.zeros( shape = frames[0].shape )
    for f in frames: summary += f
    e.images_['summary'] = to_grayscale( summary )

    get_rois( frames, window = config.n_frames)
    logger.info("Got all interesting ROIs")

    # Here we use the collected rois which are acceptable as cells and filter
    # out overlapping contours
    reallyCoolCells = get_roi_containing_minimum_cells( )
    roifile = '%s_roi.csv' % inputfile
    text = [ 'colum,row,radius' ]

    boxes = []
    for c in reallyCoolCells:
        text.append( '%s,%s,%s' % ( c.center[0], c.center[1], c.radius ))
        boxes.append( c.rectangle )
    with open( roifile, 'w' ) as f:
        f.write( '\n'.join( text ) )
    logger.info( "ROIs are written to %s " % roifile )

    # dfmat = df_by_f_data( boxes, frames )
    # e.images_['df_by_f'] = dfmat

def plot_results( ):
    outfiles = []
    stamp = datetime.datetime.now().isoformat()
    txt = "%s" % e.args_.input.split('/')[-1]
    txt += ' @ %s' % stamp
    txt += ', 1 px = %s micro-meter' % e.args_.pixal_size

    plt.figure()
    gs = matplotlib.gridspec.GridSpec(2, 2)
    ax = plt.subplot( gs[0,0] )
    ax.imshow( e.images_['summary'], aspect='equal' )
    ax.set_title( "Summary of activity in region", fontsize = 10 )

    ax = plt.subplot( gs[0,1] )
    ax.imshow(  e.images_['rois'] , aspect = 'equal'  )
    ax.set_title( 'Computed ROIs', fontsize = 10 )

    ax = plt.subplot( gs[1,0] )
    ax.imshow( 0.99*e.images_['summary']+e.images_['bound_area'], aspect='equal')
    ax.set_title( 'Clusters for df/F', fontsize = 10 )

    ax = plt.subplot( gs[1,1] )
    ax.imshow( e.images_['neurons'], aspect = 'equal' )
    ax.set_title('Filtered ROIs', fontsize = 10)
    plt.suptitle(txt , fontsize = 8)
    outfiles.append('%s_0.%s' % tuple(e.args_.output.rsplit('.', 1 )))

    if e.args_.debug:
        plt.show( )
    plt.savefig( outfiles[-1] )

    ##ax = plt.subplot(1, 1, 1)
    ##im = ax.imshow( e.images_['df_by_f'], aspect = 'auto' )
    ##ax.set_title('100*df/F in rectangle(cluster). Baseline, min() of vector' 
    ##        , fontsize = 10
    ##        )
    ##plt.colorbar( im,  orientation = 'horizontal' )
    ##if e.args_.debug:
    ##    plt.show( )
    ##plt.suptitle( txt, fontsize = 8 )
    ##outfiles.append( '%s_1.%s' % tuple(e.args_.output.rsplit('.', 1 )))
    ##plt.savefig( outfiles[-1] )

    ##logger.info('Saved results to %s' % outfiles)


def main( ):
    init( )
    process_input(  )
    plot_results( )

if __name__ == '__main__':
    import argparse
    description = '''What it does? README.md file or Ask dilawars@ncbs.res.in'''
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
    parser.add_argument('--box', '-b'
        , required = False
        , default = "0,0,-1,-1"
        , help = 'Bounding box  row1,column1,row2,column2 e.g 0,0,100,100'
        )
    parser.add_argument('--pixal_size', '-px'
        , required = False
        , type = float
        , help = 'Pixal size in micro meter'
        )
    parser.parse_args( namespace = e.args_ )
    e.args_.output = e.args_.output or ('%s_out.png' % e.args_.input)
    main( )
