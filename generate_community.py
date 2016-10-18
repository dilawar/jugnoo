#!/usr/bin/env python 

"""generate_community.py: 

TODO: 
Ideally should generate clusters but currently only generates correlation graph.

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
        if g_[s][t]['weight'] >= kwargs.get( 'minimum_correlation', 0.5 ):
            toKeep += [s,t]

    g_ = g_.subgraph( toKeep )
    return g_

def build_correlation_graph( g_, img ):
    nodes = g_.nodes( )
    corImg = np.zeros_like( img )
    corImg2 = np.zeros_like( img )
    # Since color starts with 1, we need to substract that from the index. Node
    # ids are color ids.
    for s, t in g_.edges( ):
        corImg[s-1,t-1] = g_[s][t]['weight']
        corImg2[s-1,t-1] = g_[s][t]['weight_sigma']
    return corImg, corImg2


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
    cor, corSigma = build_correlation_graph( g_, img )
    communityColor = 0
    # g_ = filter_graph( g_, minimum_correlation = 0.5 )
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


    plt.figure( figsize=(12,8) )
    gridSize = (3, 2)
    ax1 = plt.subplot2grid( gridSize, (0,0), colspan = 2 )
    ax2 = plt.subplot2grid( gridSize, (1,0), colspan = 2 )
    ax3 = plt.subplot2grid( gridSize, (2,0), colspan = 1 )
    ax4 = plt.subplot2grid( gridSize, (2,1), colspan = 1 )
    # nx.draw( g_, pos = pos )
    # h, w, d = frames.shape
    # allF = np.reshape( frames, (h*w, d) )
    im = ax1.imshow( timeseries, interpolation = 'none' ) # , aspect = 'auto' )
    plt.colorbar( im, ax = ax1 )
    newTimeSeries = []
    for x in timeseries:
        t = x[:]   # Copy else original will change
        t[ t < t.mean() + t.std() ] = 0
        t[ t >= t.mean() + t.std() ] = 1.0
        newTimeSeries.append(  t )
    im = ax2.imshow( newTimeSeries, interpolation = 'none' ) #, aspect = 'auto' )
    plt.colorbar( im, ax = ax2 )

    # plt.title( 'Firing in cells (thresholded)' )
    # plt.subplot( 122 )
    img = ax3.imshow( cor, cmap='gray', interpolation = 'none' ) #, aspect = 'auto' )
    plt.colorbar( img, ax = ax3 )
    img = ax4.imshow( corSigma, cmap = 'gray', interpolation = 'none' ) #, aspect = 'auto' )
    plt.colorbar( img, ax = ax4 )
    # ax3.set_title( 'Correlation among cells' )
    outfile = 'result.png'
    # plt.tight_layout( )
    plt.suptitle( 'Correlation among cells (raw v/s thesholded data)' )
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
