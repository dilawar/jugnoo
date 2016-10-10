#!/usr/bin/env python

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
import logging
import cv2
from collections import defaultdict

def main( npfile ):
    cells = nx.read_gpickle( graphfile )
    timeseries_ = cells.graph['timeseries']
    syncDict = defaultdict( list )
    syncImg = np.zeros( shape = cells.graph['shape'] )
    for i, c in enumerate( nx.attracting_components( cells ) ):
        if len(c) < 2:
            continue
        logging.info( 'Found attracting component of length %d' % len(c) )
        for p in c:
            cv2.circle( syncImg, (p[1], p[0]), 2, (i+1), 2 )
            syncDict[str(c)].append( cells.node[p]['timeseries'] )

    plt.subplot( 2, 1, 1)
    img = [ ]
    for k in syncDict:
        img += syncDict[k]
        print img[0].size
        img.append( np.zeros( img[0].size ) )

    plt.imshow( np.vstack( img ), interpolation = 'none', aspect = 'auto' )

    plt.subplot( 2, 1, 2 )
    plt.imshow( syncImg, interpolation = 'none', aspect = 'auto' )
    plt.savefig( 'clusters.png' )

if __name__ == '__main__':
    npfile = sys.argv[1]
    main( npfile )
