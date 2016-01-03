# Configuration file.

# Threshold value for canny edge detector. 200 is a good value. Make it -1 if
# you want the program to compute it for you. The results are very sensitive to
# this paramter.
high_threshold = 200
elow, ehigh = high_threshold/2.0, high_threshold

# Min number of points in contours. A good value is 3 - 8. Larger values reject
# the small ROIs.
min_points_in_contours = 2

# Time averaging over frames. Increase this number for motion correction. Larger
# values will give cleaner ROIs but also reduce their numbers.
n_frames = 1

# Maximum diameter (in pixals) of potential ROIs (approximated by a circle).
max_roi_diameter = 15

# Minimum diameter (in pixal ) of a ROI.
# case of doubt, make it zero.
min_roi_diameter = 5

# Canny edge detector window size for Guassian smoothening. Use 3,
canny_window_size = 3

class Args: pass
args_ = Args()
