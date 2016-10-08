#!/usr/bin/env python3

"""process_data.py: 

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
import re
import os
import matplotlib.pyplot as plt
import igraph as ig
import networkx as nx
import numpy as np
import cv2

edges_ = [ ]
nodes_ = { }
g_ = nx.Graph( )
nrows_ = 0
ncols_ = 0

def build_graph( filename, plot = False ):
    global g_
    global nodes_, nrows_, ncols_
    with open( filename, 'r' ) as f:
        lines = f.read().split( '\n' )
    edges = filter( lambda l : 'E:' in l, lines )
    for n in filter( lambda l : 'N:' in l, lines ):
        n = n.split( )
        rowIdx, colIdx = int( n[2] ), int( n[3] )
        if nrows_ <= rowIdx:
            nrows_ = rowIdx + 1
        if ncols_ <= colIdx:
            ncols_ = colIdx + 1
        nodes_[ int( n[1] ) ] = ( int(n[2]), int(n[3]) )

    for e in edges:
        e = e.split( )
        src, tgt = int( e[1] ), int( e[2] )
        w = float( e[3] )
        if w < 0.06 :
            g_.add_edge( src, tgt, weight =  w )
    print( 'No of edges %d' % g_.number_of_edges( ) )
    if plot:
        nx.draw( g_ )
        plt.savefig( 'network.png' )
    return g_

def main():
    """docstring for main"""
    g = build_graph( sys.argv[1] )
    global nodes_
    global nrows_, ncols_
    cliques = nx.k_clique_communities( g_, 5 )
    img = np.zeros( shape=(nrows_,ncols_), dtype=np.uint16 )
    for i, c in enumerate(cliques):
        pos =  [ nodes_[x] for x in c ]
        # pos = np.array( pos )
        print( pos )
        if len(pos) < 4:
            continue
        for p in pos:
            cv2.circle( img, p, 1, 10*i )
            # cv2.putText( img, str(i), p, cv2.FONT_HERSHEY_COMPLEX, 0.1, i )
        # cv2.polylines( img, np.int32( [ pos ] ), False, i ) 
    plt.imshow( img, cmap='gray', interpolation = 'none', aspect = 'auto' )
    plt.colorbar(  )
    plt.savefig( 'corr.png' )


if __name__ == '__main__':
    main()
