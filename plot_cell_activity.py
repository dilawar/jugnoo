"""plot_cell_activity.py: 

Plot activity of cells.

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
import matplotlib
matplotlib.use( 'TkAgg' )
import matplotlib.pyplot as plt
import numpy as np

def main( ):
    cells = np.load( sys.argv[1] )
    frames = np.load( sys.argv[2] )
    allSeries = []
    for i in range( int(cells.max( )) ):
        cols, rows = np.where( cells == i )
        # if len( pixals ) < 5:
            # continue 
        cellSeries = []
        for c, r in  zip(cols, rows):
            cellSeries.append( frames[r,c,:] )
        allSeries.append( np.mean( cellSeries, axis = 0 ) )
        print( 'Computing for cell with color %d' % i )
    plt.imshow( allSeries, interpolation = 'none', aspect = 'auto' )
    plt.show( )

if __name__ == '__main__':
    main()
