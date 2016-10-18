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
    # Since color starts with 1, we need to substract that from the index. Node
    # ids are color ids.
    for s, t in g_.edges( ):
        img[s-1,t-1] = g_[s][t]['weight']
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
