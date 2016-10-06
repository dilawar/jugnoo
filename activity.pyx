__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

cimport numpy as np

def build_correlation_matrix( frames ):
    cdef int m, n, i, j 
    print frames.shape
    m, n = frames.shape
    mat = np.zeros( shape=(m*n, m*n) )
    return 
