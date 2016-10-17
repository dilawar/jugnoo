#!/usr/bin/env python

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
from collections import defaultdict
import gc

import logging
logger = logging.getLogger('')

# g_ = ig.Graph( )
g_ = nx.DiGraph( )
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
        if img[p] < thres:
            return False
    return True

def build_cell_graph( graph, frames, plot = False ):
    """ Build a cell graph """
    global pixalcvs_
    global template_
    global cells_
    nodesToProcess = sorted( pixalcvs_ , reverse = True )
    cells = []
    img = np.int32( np.sum( frames, axis = 2 ))
    for (c, n) in nodesToProcess:
        if graph.node[n].get('cell', None) is not None:
            continue
        graph.node[n]['cell'] = len(cells)
        logger.debug( 'Added timeseris value to cell graph' )
        cells.append( n )
        clusters = [ frames[n[0],n[1],:] ]
        for m in graph.nodes( ):
            # Definately not part of the cell.
            if distance(m, n ) > 20.0:
                continue 
            # Accept all pixals in neighbourhood, for others check if they are
            # connected at all.
            if distance(m, n) < 10.0 or is_connected(m, n, template_):
                graph.node[m]['cell'] = len( cells )
                clusters.append( frames[m[0],m[1],:] )
        logger.info( 'Total pixal in cell %d' % len( clusters ) )
        clusters = np.mean( clusters, axis = 0 )
        cell_.add_node( n , cv = c, timeseries = clusters )

    print( '[INFO] Total cells %d' % len(cells) )
    if not plot:
        return  cell_

    for p in cells:
        cv2.circle( template_, (p[1], p[0]), 3, 20 );
    plt.figure( )
    plt.imshow( template_, interpolation = 'none', aspect = 'auto' )
    plt.colorbar( )
    plt.title( 'Total cells %d' % len(cells) )
    plt.savefig( 'cells.png' )
    return cell_

def sync_index( x, y, method = 'pearson' ):
    # Must smooth out the high frequency components.
    assert min(len( x ), len( y )) > 30, "Singal too small"
    a, b = [ smooth( x, 31 ) for x in [x, y ] ]
    if method == 'dilawar':
        signA = np.sign( np.diff( a ) )
        signB = np.sign( np.diff( b ) )
        s1 = np.sum( signA * signB ) / len( signA )
        s2 = sig.fftconvolve(signA, signB).max() / len( signA )
        return max(s1, s2) ** 0.5
    elif method == 'pearson':
        aa, bb = a / a.max(), b / b.max( )
        c = scipy.stats.pearsonr( aa, bb )
        return c[0]
    raise UserWarning( 'Method %s is not implemented' % method )

def correlate_node_by_sync( cells ):
    global template_ , avg_
    for m, n in itertools.combinations( cells.nodes( ), 2 ):
        vec1, vec2 = cells.node[m]['timeseries'], cells.node[n]['timeseries']
        corr = sync_index( vec1, vec2 )
        rcorr = sync_index( vec2, vec1 )
        if corr > 0.6:
            cells.add_edge( m, n, weight = corr )
            cells.add_edge( n, m, weight = rcorr )

    outfile = 'final.png' 
    plt.figure( figsize = (12,8) )
    plt.subplot( 2, 2, 1 )
    plt.imshow( avg_, interpolation = 'none', aspect = 'auto' )
    plt.title( 'All frames averaged' )
    plt.colorbar( ) # orientation = 'horizontal' )

    syncImg = np.zeros( shape=template_.shape )
    syncDict = defaultdict( list )
    cells.graph['shape'] = template_.shape
    cells.graph['timeseries'] = timeseries_
    nx.write_gpickle( cells, 'cells.gpickle' )
    logger.info( 'Logging out after writing to graph.' )
    return 

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
    np.save( '%s.npy' % filename, img )
    plt.savefig( filename )
    print( 'Saved figure to {0} and to {0}.npy'.format(filename ) )
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

def compute_cells( img, **kwargs ):
    """ Return dominant pixal representing a cell. 
    Start with the pixal with maximum variation.
    """
    cells = np.zeros( img.shape )
    varImg = img.copy( )

    # patch_rect_size is rectangle which represents the maximum dimension of
    # cell. We start a pixal with maximum variation and search in this patch for
    # other pixals which might be on the same cell.
    d = kwargs.get( 'patch_rect_size', 40 )
    cellColor = 1
    while True:
        (minVal, maxVal, min, x) = cv2.minMaxLoc( varImg )
        assert maxVal == varImg.max( )
        if maxVal <= img.mean():
            break

        # In the neighbourhood, find the pixals which are closer to this pixal
        # and have good variation.
        logging.info( 'Computing cell at (%3d,%3d) (%.3f)' % (x[1],x[0],maxVal))
        for i, j in itertools.product( range(d), range(d) ):
            i, j = x[1] + (i - d/2), x[0] + (j - d/2)
            if i < img.shape[0] and j < img.shape[1]:
                if is_connected( (x[1],x[0]), (i, j), img, max(maxVal - 1.0, img.mean()) ):
                    logging.debug( 'Point %d, %d is connected' % (i, j) )
                    # If only this pixal does not belong to other cell.
                    if cells[i,j] == 0:
                        cells[i, j] = cellColor
                    varImg[i, j] = 0
        cellColor += 1
    return cells

def process_input( imgfile, plot = False ):
    global g_
    global template_, avg_
    global frames_, timeseries_
    data = np.load( imgfile )
    # play( data )
    frames_ = np.dstack( [ garnish_frame( f ) for f in data.T ] )
    avg_ = np.mean( frames_, axis = 2 )
    variationAmongPixals = filter_pixals( frames_ )
    cellsImg = compute_cells( variationAmongPixals )
    if plot:
        plt.figure( figsize=( 14, 5 ) )
        plt.subplot( 1, 2, 1 )
        plt.imshow( avg_, interpolation = 'none', aspect = 'auto' )
        plt.title( 'Average activity' )
        plt.colorbar( )
        plt.subplot( 1, 2, 2 )
        print( '[INFO] Total cells %d' % cellsImg.max( ) )
        cmap = plt.get_cmap('seismic', cellsImg.max( ) )   
        plt.imshow( cellsImg, cmap = cmap, interpolation = 'none', aspect = 'auto' )
        plt.title( 'Computed cells' )
        plt.colorbar( )
        plt.tight_layout( )
        outfile = '%s.png' % imgfile
        plt.savefig( outfile )
        print( '[INFO] Wrote summary image to %s' % outfile )
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

