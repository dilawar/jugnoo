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
import networkx as nx
import numpy as np
import graph_tool as gt
import pandas

def main():
    """docstring for main"""
    g = nx.Graph( )
    data = pandas.read_csv( './data_correlation.csv' )
    r1s, c1s, r2s, c2s = data['row1'], data['col1'], data['row2'], data['col2']
    corrs = data['corr']
    N = r1s.max() * c1s.max( )
    img = np.zeros( shape=(N,N) )
    

if __name__ == '__main__':
    main()
