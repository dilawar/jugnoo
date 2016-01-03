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

import numpy as np
import environment as e
import cv2

import logging 
logger = logging.getLogger('')

def get_frame_data( frame ):
    frame = frame.convert('L')
    img = np.array(frame)
    return img

def get_bounding_box( ):
    bbox = [ int(x) for x in e.args_.box.split(',') ]
    r1, c1, h, w = bbox

    if h == -1: r2 = h
    else: r2 = r1 + h
    if w == -1: c2 = w
    else: c2 = c1 + w
    return (r1, c1, r2, c2)

def read_frames_from_avi( filename ):
    cap = cv2.VideoCapture( filename )
    frames = []
    while cap.isOpened():
        try:
            ret, frame = cap.read()
        except Exception as e:
            print("Failed to read frame. Error %s" % e)
            quit()
        if ret:
            gray = cv2.cvtColor( frame, cv2.COLOR_BGR2GRAY)
            frames.append( gray )

    logger.info("Total %s frames read" % len(frames))
    return frames


def read_frames_from_tiff( filename ):
    from PIL import Image
    tiff = Image.open( filename )
    frames = []
    try:
        i = 0
        while 1:
            i += 1
            tiff.seek( tiff.tell() + 1 )
            framedata = get_frame_data( tiff )
            bbox = get_bounding_box( )
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
    elif ext in [ 'avi', 'mp4' ]:
        return read_frames_from_avi ( videofile )
    else:
        logger.error('Format %s is not supported yet' % ext )
        quit()
