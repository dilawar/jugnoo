#!/usr/bin/env python3

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

def is_connected( m, n, img ):
    """ 
    If pixal m and pixal n have path between them, return true.
    """
    line = simg.map_coordinates(img, [ m, n ] )
    if line.min() < 1:
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

def sync_index( a, b ):
    # Must smooth out the high frequency components.
    assert min(len( a ), len( b )) > 30, "Singal too small"
    # a, b = [ smooth( x, 11 ) for x in [a, b ] ]
    signA = np.sign( np.diff( a ) )
    signB = np.sign( np.diff( b ) )
    s1 = np.sum( signA * signB ) / len( signA )
    s2 = sig.fftconvolve(signA, signB).max() / len( signA )
    return max(s1, s2)

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
    nx.write_gpickle( cells, 'cells.gpickle' )
    logger.info( 'Logging out after writing to graph.' )
    return 
    try:
        nx.drawing.nx_agraph.write_dot( cells, 'all_cell.dot' )
    except Exception as e:
        logger.warn( 'Failed to write dot file %s' % e )
    for i, c in enumerate( nx.attracting_components( cells ) ):
        if len(c) < 2:
            continue
        logger.info( 'Found attracting component of length %d' % len(c) )
        for p in c:
            cv2.circle( syncImg, (p[1], p[0]), 2, (i+1), 2 )
            # syncDict[str(c)].append( cells.node[p]['timeseries'] )

    plt.subplot( 2, 2, 2 )
    plt.imshow( timeseries_
            , interpolation = 'none', aspect = 'auto', cmap = 'seismic' )
    plt.colorbar(  ) #orientation = 'horizontal' )
    plt.title( 'Activity of each pixal' )

    plt.subplot( 2, 2, 3 )
    plt.imshow( syncImg, interpolation = 'none', aspect = 'auto' )
    plt.colorbar( ) #orientation = 'horizontal' )

    # Here we draw the synchronization.
    plt.subplot( 2, 2, 4 )
    # clusters = []
    # for c in syncDict:
        # clusters += syncDict[c]
        # # Append two empty lines to separate the clusters.
        # clusters += [ np.zeros( timeseries_.shape[1] ) ] 
    # try:
        # plt.imshow( np.vstack(clusters), interpolation = 'none', aspect = 'auto' )
        # plt.colorbar(  ) #orientation = 'horizontal' )
    # except Exception as e:
        # print( "Couldn't plot clusters %s" % e )
    plt.tight_layout( )
    plt.savefig( outfile )
    logger.info( 'Saved to file %s' % outfile )

def fix_frames( frames ):
    result = [ ]
    for f in frames:
        m, u = f.mean(), f.std( )
        # f[ f < (m - 2 * u) ] = 0
        # f[ f > (m + 2 * u) ] = 255.0
        # scipy.stats.threshold( f, m - 2*u, m + 2*u, 0, 255 )
        result.append( f )
    return np.int32( np.dstack( result ) )
        

def process_input( plot = False ):
    global g_
    global template_, avg_
    global frames_, timeseries_
    inputdir = e.args_.input
    tiffs = []
    for d, sd, fs in os.walk( inputdir ):
        for f in fs:
            if 'tiff' in f[-5:].lower() or 'tif' in f[-5:].lower():
                tiffs.append( os.path.join(d, f) )

    allFrames = []
    for inputfile in tiffs:
        logger.info("Processing %s" % inputfile)
        frames = imgr.read_frames( inputfile, min2zero = True )
        allFrames += frames 

    logger.info( 'Total frames to analyze %d' % len(allFrames ))
    template_ = np.zeros( shape = allFrames[0].shape )
    # Save the raw frames.
    raw_ = np.dstack( allFrames )

    # These are fixed frames.
    frames = fix_frames( allFrames )
    frames_ = np.int32( frames )

    # Raw average 
    avg_ = np.int32( np.mean( raw_, axis = 2 ) )

    # get all timeseries
    timeseries = []
    rows, cols = frames.shape[0:2]
    for i, j in [ (x,y) for x in range(rows) for y in range( cols ) ]:
        timeseries.append( frames[i,j,:] )
    timeseries_ = np.uint32( np.vstack( timeseries ) ) 

    for (r,c), val in np.ndenumerate( template_ ):
        pixals = frames[r,c,:]
        cv = pixals.var( ) / pixals.mean( ) 
        # if pixals.max() >= pixals.mean() + 4 * pixals.std():
        if cv > 4.0:
            template_[r,c] = pixals.mean( )
            g_.add_node( (r,c), cv = cv )
            pixalcvs_.append( (cv, (r,c) ) )
    if plot:
        plt.subplot(2, 1, 1 )
        plt.imshow( template_ )
        plt.colorbar( )
        plt.subplot(2, 1, 2 )
        plt.imshow( avg_ )
        plt.tight_layout( )
        plt.colorbar( )
        plt.savefig( "template.png" )
        print( 'Saved to template.png' )

    cells = build_cell_graph( g_, frames )
    correlate_node_by_sync( cells )

    
def main( ):
    t1 = time.time()
    process_input(  )
    print( '[INFO] Total time taken %f seconds' % (time.time() - t1) )

if __name__ == '__main__':
    import argparse
    description = '''What it does? README.md file or Ask dilawars@ncbs.res.in'''
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--input', '-i'
        , required = True
        , help = 'Input dir'
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



