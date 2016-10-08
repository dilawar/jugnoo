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

nodes_ = [ ]
g_ = ig.Graph( )

def add_node( nodename ):
    nodeId = 0
    if nodename not in nodes_:
        nodes_.append( nodename )
        nodeId = len( nodes_ )
        g_.add_vertex( nodeId, label = nodename )
        return nodeId
    nodeId = nodes_.index( nodename )
    return nodeId


def build_graph( filename ):
    with open( filename, 'r' ) as f:
        lines = f.read().split( '\n' )
    edges = filter( lambda l : '->' in l, lines )
    edgePat = re.compile( r'(?P<src>.+?)->(?P<tgt>.+?)\[(?P<attr>.*?)\]', re.DOTALL )
    for e in edges:
        m = edgePat.search( e )
        src, tgt = m.group( 'src' ), m.group( 'tgt' )
        srcId, tgtId = add_node( src ), add_node( tgt )
        attr = m.group( 'attr' )
    print( 'Total nodes %d' % len( nodes_ ) )

def main():
    """docstring for main"""
    g = build_graph( sys.argv[1] )
    print( g )


if __name__ == '__main__':
    main()
