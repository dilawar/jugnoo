"""image_reader.py: 

    Extract frames from input file.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import logging 
logger = logger.getLogger('')

def read_frames_from_tiff( filename ):
    from PIL import Image
    tiff = Image.open( tiff_file )
    try:
        i = 0
        while 1:
            i += 1
            tiff.seek( tiff.tell() + 1 )
            framedata = get_frame_data( tiff )
            if bbox:
                framedata = framedata[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            frames.append( framedata )
    except EOFError as e:
        logger.info("Total frames: %s" % i )
        logger.info("All frames are processed")
    return frames

def read_frames( videofile ):
    ext = videofile.split('.')[-1]
    if ext in [ 'tif', 'tiff' ]:
        return read_frames_from_tiff( videofile )
    else:
        logger.error('Format %s is not supported yet' )
        quit()
