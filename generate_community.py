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

def filter_graph( g_, **kwargs ):
    toKeep, listEdges = [], []
    for s, t in g_.edges():
        listEdges.append( (g_[s][t]['corr'], s, t ) )
    largeCorrEdges = sorted( listEdges
            , reverse = True)[0:kwargs.get('maximum_edges', 500 )]
    for (c,s,t) in largeCorrEdges:
        toKeep += [s,t]
    g_ = g_.subgraph( toKeep )
    return g_


def main( **kwargs ):
    graph = kwargs[ 'community' ]
    frames = np.load( kwargs['frames'] )

    if isinstance( graph, str):
        g_ = nx.read_gpickle( graph )
    else:
        g_ = graph

    img = np.zeros( shape=g_.graph['shape'] )
    print( '[INFO] Dims of community image %s' % str(img.shape))
    communityColor = 0
    print( '[INFO] Total edges in graph %d' % g_.number_of_edges( ) )
    g_ = filter_graph( g_, maximum_edges = 500 )
    print( '[INFO]  Total edges in graph (post filter) %d' % g_.number_of_edges())
    timeseries = []
    for k in nx.k_clique_communities( g_, 2 ):
        print( 'Found a community : %s' % k )
        communityColor += 1
        for cell in k:
            indices = g_.node[cell]['indices']
            tvec = g_.node[cell]['timeseries']
            timeseries.append( tvec / tvec.max( ) )
            for i, j in indices:
                img[j,i] = communityColor

    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout( g_, 'neato' )
    plt.figure( figsize=(12,5) )
    plt.subplot( 121 )
    # nx.draw( g_, pos = pos )
    # h, w, d = frames.shape
    # allF = np.reshape( frames, (h*w, d) )
    plt.imshow( timeseries, interpolation = 'none', aspect = 'auto' )
    plt.colorbar( )
    plt.subplot( 122 )
    plt.imshow( img, interpolation = 'none', aspect = 'auto' )
    plt.colorbar( )
    outfile = 'result.png'
    plt.savefig( outfile )
    print( '[INFO] Saved results to %s' % outfile )


if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Generate cliques out of community graph'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--community', '-c'
        , required = True
        , help = 'Input community graph (pickle)'
        )
    parser.add_argument('--frames', '-f'
        , required = True
        , help = 'Numpy stack of all frames'
        )
    parser.add_argument('--output', '-o'
        , required = False
        , help = 'Output file'
        )
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    main( **vars( args ) )
