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
import cell

import logging
logger = logging.getLogger('')

def init( ):
    e.save_direc_ = os.path.join( '.', '_results_%s' % os.path.split(e.args_.file)[1])
    if os.path.isdir( e.save_direc_ ):
        return
        for f in glob.glob( '%s/*' % e.save_direc_ ):
            os.remove(f)
    else:
        logging.info("Creating dir %s for saving data" % e.save_direc_)
        os.makedirs( e.save_direc_ )

def get_frame_data( frame ):
    frame = frame.convert('L')
    img = np.array(frame)
    return img

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
            e = threshold_frame( f, nstd = 2)
            sumAll += e
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
    global cells_
    # First fit it with an ellipse
    cell = cell.Cell( contour )
    if cell.area < config.min_neuron_area:
        logger.debug("Rejected contour because cell area was too low")
        return False

    if cell.area > config.max_neuron_area:
        logger.debug(
                "Rejected contour %s because of its area=%s" % (contour, cell.area)
            )
        return False

    # If the lower axis is 0.7 or more times of major axis, then aceept it.
    if cell.eccentricity < 0.7:
        msg = "Contour %s is rejected because " % contour
        msg += "axis ration (%s) of cell is too skewed" % cell.eccentricity
        logger.debug( msg )
        return False

    cells_.append( cell )
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

def df_by_f( roi, frames ):
    logger.info( "ROI: %s" % str(roi) )
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
        vec = df_by_f( r, frames )
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

    # Get cells with with area in sweat range : 10 - 12 um diameter.
    # keypoins = [ cv2.KeyPoint(c.center[0], c.center[1], c.radius ) for c in cells ]
    cells = helper.remove_duplicates( cells )

    # Now remove all those cells which are inside another cells. They should be
    # close enough in size.
    cells = helper.remove_contained_cells( cells )
    return cells

def get_roi_containing_minimum_cells( ):
    global e.images_

    neuronImg = np.zeros( shape = e.shape_ )
    coolcells = []
    for cell in cells_:
        # If area of contour is too low, reject it.
        area = cell.area
        if area < config.min_neuron_area:
            continue
        coolcells.append( cell )

    # Now we need reject some of these rectangles.
    coolerCells = merge_or_reject_cells( coolcells )

    # Replace boxex with cells later.
    boxes = []
    for c in coolerCells:
        (x, y), r = c.circle
        cr = cv2.circle( neuronImg, (int(x), int(y)), int(r) , 255, 1)
        boxes.append( c.rectangle )
    e.images_['neurons'] = neuronImg
    return set(boxes)

def process_input( inputfile, bbox = None ):
    logger.info("Processing %s" % inputfile)
    frames = read_frames( inputfile )
    
    # get the summary of all activity
    summary = np.zeros( shape = frames[0].shape )
    for f in frames: summary += f
    e.images_['summary'] = to_grayscale( summary )

    get_rois( frames, window = config.n_frames)

    # Here we use the collected rois which are acceptable as cells and filter
    # out overlapping contours
    boxes = get_roi_containing_minimum_cells( )

    dfmat = df_by_f_data( boxes, frames )
    e.images_['df_by_f'] = dfmat

def plot_results( ):

    ax = plt.subplot(3, 2, 1)
    ax.imshow( e.images_['summary'] )
    ax.set_title( "Summary of activity in region", fontsize = 10 )

    ax = plt.subplot(3, 2, 2)
    ax.imshow(  e.images_['rois'] )
    ax.set_title( 'Computed ROIs', fontsize = 10 )

    ax = plt.subplot(3, 2, 3)
    ax.imshow( 0.99*e.images_['summary'] + e.images_['bound_area'] )
    ax.set_title( 'Clusters for df/F', fontsize = 10 )

    ax = plt.subplot(3, 2, 4)
    ax.imshow( e.images_['neurons'] )
    ax.set_title('Maximal set of ROIs', fontsize = 10)

    ax = plt.subplot( 3, 1, 3, frameon=False ) 
    im = ax.imshow( e.images_['df_by_f'] )
    ax.set_title( '100*df/F in rectangle(cluster). Baseline, min() of vector' 
            , fontsize = 10
            )
    plt.colorbar( im, orientation = 'horizontal' )

    stamp = datetime.datetime.now().isoformat()

    txt = "%s" % e.args_.file.split('/')[-1]
    txt += ' @ %s' % stamp
    txt += ', 1 px = %s micro-meter' % e.args_.pixal_size
    plt.suptitle(txt
            , fontsize = 8
            , horizontalalignment = 'left'
            , verticalalignment = 'bottom' 
            )
    plt.tight_layout( 1.5 )
    logger.info('Saved results to %s' % e.args_.outfile)
    if e.args_.debug:
        plt.show( )
    plt.savefig( e.args_.outfile )

def get_bounding_box( ):
    bbox = [ int(x) for x in e.args_.box.split(',') ]
    r1, c1, h, w = bbox

    if h == -1: r2 = h
    else: r2 = r1 + h
    if w == -1: c2 = w
    else: c2 = c1 + w
    return (r1, c1, r2, c2)

def main( inputfile, **kwargs ):
    init( **kwargs )
    bbox = get_bounding_box( )
    logger.info("== Bounding box: %s" % str(bbox))
    process_input( e.args_.input, bbox = bbox )
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
        , required = True
        , type = float
        , help = 'Pixal size in micro meter'
        )
    class Args(): pass
    parser.parse_args( namespace = e.args_ )
    e.args_.outfile = args.outfile or ('%s_out.png' % args.file)
    main( e.args_.input, **vars(e.args_) )
