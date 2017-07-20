#!/usr/bin/env python

"""


"""

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
import scipy
import scipy.ndimage as simg
import os
import sys
import time
import cv2
import image_reader as imgr
import environment as e
import networkx as nx
import itertools
import random
from collections import defaultdict
import gc

import logging
logger = logging.getLogger('')

# g_ = ig.Graph( )
g_ = nx.Graph( )
cell_ = nx.DiGraph( ) 
indir_ = None
pixalcvs_ = []
template_ = None

# Keep the average of all activity here 
avg_ = None

timeseries_ = None
frames_ = None
raw_ = None

def plot_two_pixals( a, b, window = 3 ):
    plt.figure( )
    smoothW = np.ones( window ) / window 
    pixalA, pixalB = [ smooth( x ) for x in [a, b] ]
    plt.subplot( 2, 1, 1 )
    plt.plot( pixalA )
    plt.plot( pixalB )
    plt.subplot( 2, 1, 2 )
    plt.plot( convolve( pixalA, pixalB ) )
    plt.show( )

def smooth( a, window = 3 ):
    window = np.ones( window ) / window 
    return np.convolve( a, window , 'same' )

def distance( p1, p2 ):
    x1, y1 = p1
    x2, y2 = p2 
    return (( x1 - x2 ) ** 2.0 + (y1 - y2 ) ** 2.0 ) ** 0.5

def is_connected( m, n, img, thres = 10 ):
    """ 
    If pixal m and pixal n have path between them, return true.

    Make sure that m and n are bound within image.
    """
    if n[0] != m[0]:
        slope = float(n[1] - m[1]) / float(n[0] - m[0])
        if n[0] < m[0]:
            x = np.arange(m[0], n[0]-1, -1 )
        else:
            x = np.arange(m[0], n[0]+1, 1 )
        points = [ (a, int(m[1] + (a - m[0]) * slope)) for a in x ]
    else:
        if n[1] > m[1]:
            ys = np.arange(m[1], n[1]+1, 1 )
        else:
            ys = np.arange(m[1], n[1]-1, -1 )
        points = [ (m[0], y) for y in ys ]

    for p in points:
        x, y = int(p[0]), int(p[1])
        if img[x,y] < thres:
            return False
    return True

def sync_index( x, y, method = 'pearson' ):
    # Must smooth out the high frequency components.
    assert min(len( x ), len( y )) > 30, "Singal too small"
    a, b = [ smooth( x, 31 ) for x in [x, y ] ]
    coef = 0.0
    if method == 'dilawar':
        signA = np.sign( np.diff( a ) )
        signB = np.sign( np.diff( b ) )
        s1 = np.sum( signA * signB ) / len( signA )
        s2 = sig.fftconvolve(signA, signB).max() / len( signA )
        coef = max(s1, s2) ** 0.5
    elif method == 'pearson':
        aa, bb = a / a.max(), b / b.max( )
        c = scipy.stats.pearsonr( aa, bb )
        coef = c[0]
    else:
        raise UserWarning( 'Method %s is not implemented' % method )
    # print( '\tCorr coef is %f' % coef )
    return coef


def fix_frames( frames ):
    result = [ ]
    for f in frames:
        m, u = f.mean(), f.std( )
        # f[ f < (m - 2 * u) ] = 0
        # f[ f > (m + 2 * u) ] = 255.0
        result.append( f )
    return np.int32( np.dstack( result ) )

def garnish_frame( frame ):
    """ Test modification to frame.
    Doesn't work well because some patches of images are pretty bright.
    """
    # f = cv2.medianBlur( frame, 3 )
    # f = cv2.GaussianBlur( frame, (5,5), 0 )
    f = cv2.bilateralFilter( frame, 5, 50, 50 )
    return f

def save_image( img, filename, **kwargs ):
    plt.figure( )
    plt.imshow( img, interpolation = 'none', aspect = 'auto' )
    plt.colorbar( )

    if kwargs.get( 'title', False):
        plt.title( kwargs.get( 'title' ) )

    # Saving to numpy format.
    np.save( '%s.npy' % filename, img )
    plt.savefig( filename )
    print( '[INFPO] Saved figure {0}.png and data to {0}'.format(filename ) )
    plt.close( )

def play( img ):
    for f in img.T:
        cv2.imshow( 'frame', np.hstack((f, garnish_frame(f))) )
        cv2.waitKey(10)

def filter_pixals( frames, plot = False ):
    r, c, nframes = frames.shape
    cvs = np.zeros( shape = (r,c) )
    for i, j in itertools.product( range(r), range(c) ):
        pixals = frames[i,j,:]
        cv = pixals.var() / pixals.mean()
        cvs[i,j] = cv 
    # Now threshold the array at mean + std. Rest of the pixals are good to go.
    cvs = scipy.stats.threshold( cvs, cvs.mean( ) + cvs.std( ), cvs.max(), 0 )
    if plot:
        save_image( cvs, 'variations.png', title ='Variation in pixal' )
    return cvs

def compute_cells( variation_img, **kwargs ):
    """ Return dominant pixal representing a cell. 
    Start with the pixal with maximum variation.
    """

    cells = np.zeros( variation_img.shape )
    varImg = variation_img.copy( )

    # patch_rect_size is rectangle which represents the maximum dimension of
    # cell. We start a pixal with maximum variation and search in this patch for
    # other pixals which might be on the same cell.
    d = kwargs.get( 'patch_rect_size', 40 )

    breakAt = varImg.mean()
    cellColor = 0
    while True:
        (minVal, maxVal, min, x) = cv2.minMaxLoc( varImg )
        # Assign random color value.
        #cellColor += 1
        cellColor = random.randint(0, 32)
        assert maxVal == varImg.max( )
        if maxVal <= breakAt:
            break

        # In the neighbourhood, find the pixals which are closer to this pixal
        # and have good variation. It might belog to same cell.
        print( '+  Cell at (%3d,%3d) (var: %.3f)' % (x[1],x[0],maxVal))
        for i, j in itertools.product( range(d), range(d) ):
            i, j = x[1] + (i - d/2), x[0] + (j - d/2)
            if i < variation_img.shape[0] and j < variation_img.shape[1]:
                if is_connected( (x[1],x[0]), (i, j), variation_img, max(maxVal - 1.0, variation_img.mean()) ):
                    logging.debug( 'Point %d, %d is connected' % (i, j) )
                    # If only this pixal does not belong to other cell.
                    i, j = int(i), int(j)
                    if cells[i,j] == 0.0:
                        cells[i, j] = cellColor
                    # Make this pixal to zero so it doesn't appear in search for
                    # max again.
                    varImg[i, j] = 0
    # Reverse the cells colors, it helps when plotting. 
    cells =  np.uint32( 1 + cells.max() - cells )
    print( '[INFO] Done locating all cells' )
    return cells

def threshold_signal( x ):
    v = x.copy( )
    v[ v < v.mean() + v.std() ] = 0
    v[ v >= v.mean() + v.std() ] = 1.0
    return v

def sync_index_clip( v1, v2 ):
    a = threshold_signal( v1 )
    b = threshold_signal( v2 )
    coef = 0.0
    c = scipy.stats.pearsonr( a, b )
    coef = c[0]
    return coef


def activity_in_cells( cells, frames ):
    allActivity = []
    #  This dictionary keeps the location of cells and average activity in each
    #  cell.
    global g_
    g_.graph['shape'] = cells.shape
    goodCells = {}
    for cellColor in range(1, int( cells.max( ) ) ):
        print( '+ Computing for cell color %d' % cellColor )
        xs, ys = np.where( cells == cellColor )
        if len(xs) < 1 or len(ys) < 1:
            continue
        pixals = list( zip( xs, ys )) # These pixals belong to this cell.
        if len( pixals ) < 1:
            continue

        cellActivity = []
        g_.add_node( cellColor )
        g_.node[cellColor]['pixals'] = pixals
        for x, y  in pixals:
            cellActivity.append( frames[y,x,:] )
        cellVec = np.mean( cellActivity, axis = 0 ) 
        g_.node[cellColor][ 'activity' ] = cellVec 
        # Attach this activity to graph as well after normalization.
        allActivity.append( cellVec / cellVec.max( ) )

    # Now compute correlation between nodes and add edges
    for n1, n2 in itertools.combinations( g_.nodes( ), 2):
        v1, v2 = g_.node[n1]['activity'], g_.node[n2]['activity']
        g_.add_edge( n1, n2
                , weight = sync_index( v1, v2, 'dilawar' ) 
                , weight_sigma = sync_index_clip( v1, v2 )
                )
    cellGraph = 'cells_as_graph.gpickle'
    nx.write_gpickle( g_, cellGraph )
    print( '[INFO] Wrote cell graph to pickle file %s' % cellGraph )
    print( '\t nodes %d' % g_.number_of_nodes( ) )
    print( '\t edges %d' % g_.number_of_edges( ) )
    activity = np.vstack( allActivity )
    return activity 

def process_input( imgfile, plot = False ):
    global g_
    global template_, avg_
    global frames_, timeseries_
    data = np.load( imgfile )

    ## Play the read file here.
    #play( data )

    frames_ = np.dstack( [ garnish_frame( f ) for f in data.T ] )
    print( '[INFO] Total frames read %d' % len( frames_ ))

    avg_ = np.mean( frames_, axis = 2 )
    variationAmongPixals = filter_pixals( frames_ )

    cellsImg = compute_cells( variationAmongPixals )
    activity = activity_in_cells( cellsImg, data )

    plt.figure( figsize=(14, 8) )
    ax1 = plt.subplot2grid( (2,2), (0, 0) )
    ax2 = plt.subplot2grid( (2,2), (0, 1) )
    ax3 = plt.subplot2grid( (2,2), (1, 0), colspan=2)

    if plot:
        img = ax1.imshow( avg_, cmap = 'gray', interpolation = 'none', aspect = 'auto' )
        ax1.set_title( 'Average activity' )
        plt.colorbar( img, ax = ax1 )
        print( '[INFO] Total cells %d' % cellsImg.max( ) )
        img = ax2.imshow( cellsImg, interpolation = 'none', aspect = 'auto' )
        ax2.set_title( 'Computed ROIs (cells)' )
        plt.colorbar( img, ax = ax2 )

        # img = ax3.imshow( activity, interpolation = 'none', aspect = 'auto' )
        img = ax3.imshow( activity, cmap='gray', aspect = 'auto' )
        ax3.set_title( 'Acitivity in ROIs (cells)' )
        plt.colorbar( img, ax = ax3 )
        outfile = '%s.png' % imgfile
        plt.tight_layout( )
        plt.savefig( outfile )
        print( '[INFO] Wrote computed cells to  %s' % outfile )
    return cellsImg

    
def main( args ):
    t1 = time.time()
    imgfile = args.input
    cells = process_input( imgfile, True  )
    print( '[INFO] Total time taken %f seconds' % (time.time() - t1) )
    outfile = 'cells.npy' or '%s.npy' % imgfile 
    np.save( outfile, cells )
    print( 'Wrote computed cells to %s' % outfile )


if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Compute cells in recording.'''
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
    main( args )

