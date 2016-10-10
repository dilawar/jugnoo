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

def main( graph ):
    if isinstance( graph, str):
        g_ = nx.read_gpickle( graph )
    else:
        g_ = graph

    img = np.zeros( shape=g_.graph['shape'] )
    communityColor = 0
    for k in nx.k_clique_communities( g_, 1 ):
        print( 'Found a community : %s' % k )
        communityColor += 1
        for cell in k:
            indices = g_.node[cell]['indices']
            for i, j in indices:
                print( i, j )
                img[i,j] = communityColor

    from networkx.drawing.nx_agraph import graphviz_layout
    pos = graphviz_layout( g_, 'neato' )
    plt.figure( )
    plt.subplot( 2, 1, 1 )
    nx.draw( g_, pos = pos )
    plt.savefig( 'clusters.png' )

    plt.figure( )
    plt.imshow( img, interpolation = 'none', aspect = 'auto' )
    plt.colorbar( )
    plt.savefig( 'result.png' )


if __name__ == '__main__':
    graph = sys.argv[1]
    main( graph )
