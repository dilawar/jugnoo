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

def main( graphfile ):
    g = nx.read_gpickle( graphfile )
    for i, c in enumerate( nx.attracting_components( cells ) ):
        if len(c) < 2:
            continue
        logger.info( 'Found attracting component of length %d' % len(c) )
        for p in c:
            cv2.circle( syncImg, (p[1], p[0]), 2, (i+1), 2 )
            # syncDict[str(c)].append( cells.node[p]['timeseries'] )

    plt.subplot( 2, 2, 2 )
    plt.imshow( timeseries_
            , interpolation = 'none', aspect = 'auto', cmap = 'seismic' )
    plt.colorbar(  ) #orientation = 'horizontal' )
    plt.title( 'Activity of each pixal' )

    plt.subplot( 2, 2, 3 )
    plt.imshow( syncImg, interpolation = 'none', aspect = 'auto' )
    plt.colorbar( ) #orientation = 'horizontal' )

    # Here we draw the synchronization.
    plt.subplot( 2, 2, 4 )
    # clusters = []
    # for c in syncDict:
        # clusters += syncDict[c]
        # # Append two empty lines to separate the clusters.
        # clusters += [ np.zeros( timeseries_.shape[1] ) ] 
    # try:
        # plt.imshow( np.vstack(clusters), interpolation = 'none', aspect = 'auto' )
        # plt.colorbar(  ) #orientation = 'horizontal' )
    # except Exception as e:
        # print( "Couldn't plot clusters %s" % e )
    plt.tight_layout( )
    plt.savefig( outfile )
    logger.info( 'Saved to file %s' % outfile )

if __name__ == '__main__':
    graphfile = sys.argv[1]
    main( graphfile )
