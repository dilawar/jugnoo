#!/usr/bin/env python3

__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2016, Dilawar Singh"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import matplotlib
# matplotlib.use( 'TkAgg' )
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import scipy.signal
import itertools

import logging
logging.basicConfig( level = logging.INFO )
logger = logging.getLogger( 'correlation' )

# g_ = nx.DiGraph( )
g_ = nx.Graph( )

def smooth( a, window = 3 ):
    window = np.ones( window ) / window 
    return np.convolve( a, window , 'same' )

def sync_index( x, y, method = 'pearson' ):
    # Must smooth out the high frequency components.
    assert min(len( x ), len( y )) > 30, "Singal too small"
    a, b = [ smooth( x, 31 ) for x in [x, y ] ]
    if method == 'dilawar':
        signA = np.sign( np.diff( a ) )
        signB = np.sign( np.diff( b ) )
        s1 = np.sum( signA * signB ) / len( signA )
        s2 = scipy.signal.fftconvolve(signA, signB).max() / len( signA )
        return max(s1, s2) ** 0.5
    elif method == 'pearson':
        aa, bb = a / a.max(), b / b.max( )
        c = scipy.stats.pearsonr( aa, bb )
        return c[0]
    raise UserWarning( 'Method %s is not implemented' % method )


def create_correlate_graph( graph ):
    global g_
    nodes = graph.nodes()
    for n1, n2 in itertools.combinations( nodes, 2 ):
        t1 = graph.node[n1]['timeseries']
        t2 = graph.node[n2]['timeseries']
        s = sync_index( t1, t2, 'dilawar' )
        if s > 0.75:
            logger.debug( "Edge %s -> %s (%.3f)" % (n1, n2, s ) )
            g_.add_edge( n1, n2, corr = s )
            # g_.add_edge( n2, n1, corr = s )

    # outdeg = g_.out_degree()
    outdeg = g_.degree()
    to_keep = [ n for n in outdeg if outdeg[n] > 1 ]
    g_ = g_.subgraph( to_keep )


def main( **kwargs ):
    cells = kwargs['cells']
    frames = kwargs['frames']
    if isinstance( cells, str):
        cells = np.load( cells )
    if isinstance( frames, str):
        frames = np.load( frames )

    logger.info( 'Creating correlation graph' )
    N = int( cells.max() )
    for i in range(1, N):
        logger.info( '\tDone %d out of %d' % (i, N) )
        indices = list(zip( *np.where( cells == i ) ))
        if len( indices ) < 2:
            continue
        pixals = []
        for y, x in indices:
            pixals.append( frames[x,y,:] )
        pixals = np.mean( pixals , axis = 0)
        g_.add_node( i, timeseries = pixals, indices = indices )

    g_.graph[ 'shape' ] = frames[:,:,0].shape
    create_correlate_graph( g_ )
    outfile = kwargs.get( 'output', False)  or  'correlation_graph.pickle'
    logger.info( 'Writing pickle of graph to %s' % outfile )
    nx.write_gpickle( g_, outfile )
    logger.info( 'Graph pickle is saved to %s' % outfile )


if __name__ == '__main__':
    # It accepts two files.
    import argparse
    # Argument parser.
    description = '''Generate correlation graph.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cells', '-c'
        , required = True
        , help = 'Numpy file representing cells.'
        )
    parser.add_argument('--frames', '-f'
        , required = True
        , help = 'A numpy file containing timeseries of every pixal'
        )
    parser.add_argument('--output', '-o'
        , required = False
        , help = 'Outfile graph file (pickle)'
        )
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    main( **vars(args) )
