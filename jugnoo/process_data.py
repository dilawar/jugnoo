#!/usr/bin/env python

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
import os
import matplotlib.pyplot as plt
import igraph as ig
import numpy as np
import pandas

def main():
    """docstring for main"""
    g = ig.Graph( )
    data = pandas.read_csv( './data_correlation.csv' )
    r1s, c1s, r2s, c2s = data['row1'], data['col1'], data['row2'], data['col2']
    corrs = data['corr']
    # N = r1s.max() * c1s.max( )
    # img = np.zeros( shape=(N,N) )
    for i, r in enumerate( r1s ):
        if i % 10000 == 0:
            print(i, len(r1s) )
        a, b = (c1s[i], r1s[i]), ( r2s[i], c2s[i]) 
        n1 = g.add_vertex( a )
        n2 = g.add_vertex( b )
        g.add_edge(n1, n2, weight = corrs[i] )

if __name__ == '__main__':
    main()
