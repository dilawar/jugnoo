#!/usr/bin/env python 

"""generate_community.py: 

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
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import itertools
import scipy.signal as sig

def filter_graph( g_, **kwargs ):
    toKeep, listEdges = [], []
    for s, t in g_.edges():
        if g_[s][t]['corr'] >= kwargs.get( 'minimum_correlation', 0.5 ):
            toKeep += [s,t]

    g_ = g_.subgraph( toKeep )
    return g_

def smooth( a, window = 3 ):
    window = np.ones( window ) / window 
    return np.convolve( a, window , 'same' )

def sync_index( x, y, method = 'pearson' ):
    # Must smooth out the high frequency components.
    assert min(len( x ), len( y )) > 30, "Singal too small"
    a, b = [ smooth( x, 11 ) for x in [x, y ] ]
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

def build_correlation_graph( g_, img ):
    nodes = g_.nodes( )
    for m, n in itertools.combinations( nodes, 2 ):
        vec1 = g_.node[ m ]['activity']
        vec2 = g_.node[ n ]['activity']
        corr = sync_index( vec1, vec2, 'dilawar' )
        g_.add_edge( m, n, corr = corr )
        img[m, n] = corr
    return img


def main( **kwargs ):
    graph = kwargs[ 'cell_graph' ]
    if isinstance( graph, str):
        g_ = nx.read_gpickle( graph )
    else:
        g_ = graph


    print( '[INFO] Total nodes in graph %d' % g_.number_of_nodes( ) )
    print( '[INFO] Total edges in graph %d' % g_.number_of_edges( ) )
    n = g_.number_of_nodes( )
    img = np.zeros( shape=(n,n) )
    img = build_correlation_graph( g_, img )
    communityColor = 0
    g_ = filter_graph( g_, minimum_correlation = 0.5 )
    print( '[INFO]  Total edges in graph (post filter) %d' % g_.number_of_edges())

    # Compute minimum cut.
    # res = nx.minimum_cut( g_, 0,  100, capacity = 'corr' )
    # res = nx.current_flow_closeness_centrality( g_, weight='corr' )
    # print( res )

    timeseries = []
    for k in nx.find_cliques( g_ ):
        print( 'Found a community : %s' % k )
        communityColor += 1
        for cell in k:
            indices = g_.node[cell]['pixals']
            tvec = g_.node[cell]['activity']
            timeseries.append( tvec / tvec.max( ))


    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout( g_, 'neato' )
    plt.figure( figsize=(12,5) )
    plt.subplot( 121 )
    # nx.draw( g_, pos = pos )
    # h, w, d = frames.shape
    # allF = np.reshape( frames, (h*w, d) )
    newTimeSeries = []
    for t in timeseries:
        t[ t < t.mean() + t.std() ] = 0
        t[ t >= t.mean() + t.std() ] = 1.0
        newTimeSeries.append(  t )
    plt.imshow( newTimeSeries, interpolation = 'none', aspect = 'auto' )
    plt.title( 'Firing in cells (thresholded)' )
    plt.colorbar( )
    plt.subplot( 122 )
    plt.imshow( img, cmap='gray', interpolation = 'none', aspect = 'auto' )
    plt.title( 'Correlation among cells' )
    plt.colorbar( )
    outfile = 'result.png'
    plt.savefig( outfile )
    print( '[INFO] Saved results to %s' % outfile )


if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Generate cliques out of community graph'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cell-graph', '-c'
        , required = True
        , help = 'Input community graph (pickle)'
        )
    parser.add_argument('--output', '-o'
        , required = False
        , help = 'Output file'
        )
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    main( **vars( args ) )
