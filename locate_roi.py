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
import matplotlib.pyplot as plt
import scipy.signal as sig
import os
import sys
import time
import cv2
import environment as e
import image_reader as imgr
import igraph as ig

import logging
logger = logging.getLogger('')

g_ = ig.Graph( )
indir_ = None

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

def convolve( a, b ):
    """ Just convolve begining of the signal """
    full = sig.convolve( a[:20], b[:20], mode = 'full' )
    return full[len(full)/2:]

def activity_corr( a, b ):
    N = 10
    self = convolve( a, a )[0:N]
    ab = convolve( a, b )[0:N]
    return np.sum( self / ab ) / N

def build_correlation_graph( graph, frames ):
    logger.info( 'Total pixals %d' % ( graph.vcount() ** 2 ))
    for i, m in enumerate(graph.vs( )):
        logger.info( 'Done nodes %d (out of %d)' % (i, graph.vcount() ) )
        for n in graph.vs( ):
            mname, nname = m['name'], n['name']
            if m == n: 
                continue 
            (m1, m2), (n1, n2) = eval(mname), eval(nname)
            pixalsM, pixalsN = frames[m1,m2,:], frames[n1,n2,:]
            # plot_two_pixals( pixalsM, pixalsN )
            corr = activity_corr( pixalsM, pixalsN )
            graph.add_edge( m, n, weight=corr )
            logger.debug( 'Computed corr between %s and %s = %s' % ( 
                mname, nname, corr) 
                )
    logger.info( 'Done building graph. Now extract communities' )
    outfile = os.path.join( e.args_.input, 'activity_correlation.gml' )
    ig.write( graph, outfile, format = 'graphml' )
    print( '[INFO] Wrote graph to %s' % outfile )

def process_input( ):
    global g_
    inputdir = e.args_.input
    tiffs = []
    for d, sd, fs in os.walk( inputdir ):
        for f in fs:
            if 'tiff' in f[-5:].lower() or 'tif' in f[-5:].lower():
                tiffs.append( os.path.join(d, f) )

    allFrames = []
    for inputfile in tiffs:
        logger.info("Processing %s" % inputfile)
        frames = imgr.read_frames( inputfile )
        allFrames += frames 

    template = np.zeros( shape = allFrames[0].shape )
    frames = np.dstack( allFrames )
    for i in range(frames.shape[2]):
        f = frames[:,:,i]
    for (r,c), val in np.ndenumerate( template ):
        pixals = frames[r,c,:]
        if pixals.var( ) > 300.0:
            template[r,c] = pixals.mean( )
            g_.add_vertex( name=str((r,c)) )

    build_correlation_graph( g_, frames )
    plt.subplot(2, 1, 1 )
    plt.imshow( template )
    plt.colorbar( )
    plt.subplot(2, 1, 2 )
    plt.imshow( np.sum( frames, axis = 2 ) )
    plt.savefig( "template.png" )
    print( 'Saved to template.png' )

    
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
