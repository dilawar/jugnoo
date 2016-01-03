"""neuron.py: 

    A class representing neuron.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2015, Dilawar Singh and NCBS Bangalore"
__credits__          = ["NCBS Bangalore"]
__license__          = "GNU GPL"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import cv2 
import environment as e
import logging
logger = logging.getLogger( '' )

class Cell():
    def __init__(self, contour):
        self.contour = contour
        self.rectangle = cv2.boundingRect( contour )
        self.circle = cv2.minEnclosingCircle( contour )
        self.area = cv2.contourArea( contour ) * (e.args_.pixal_size ** 2.0)
        if len(contour) > 5:
            self.geometry = cv2.fitEllipse( contour )
            self.geometry_type = 'ellipse'
            axis = self.geometry[1]
            self.eccentricity = axis[0]/axis[1]
            self.radius = sum(axis) / 2.0
        else:
            self.geometry = cv2.minEnclosingCircle( contour )
            self.geometry_type = 'circle'
            self.eccentricity = 1.0
            self.radius = self.geometry[1]
            
        self.center = self.geometry[0]
        logging.debug("A potential cell : %s" % self)

    def __repr__(self):
        return "Center: %s, area: %s" % (self.center, self.area )


