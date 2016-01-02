"""helper.py: 

Helper functions.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import pylab

def plot_images( images_dict, outfile = None ):
    assert type( images_dict ) == dict, "Requires dict of images"
    fig, axes = pylab.subplots( len(images_dict), 1, sharex = True )
    for i, k in enumerate( images_dict ):
        axes[i].imshow( images_dict[k] )
        axes[i].set_title( k )
    if not outfile:
        pylab.show( )
    else:
        print('[INFO] Saving image to %s' % outfile )
        pylab.savefig( outfile )
