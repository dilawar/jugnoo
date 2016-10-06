#!/bin/bash - 
#===============================================================================
#
#          FILE: run.sh
# 
#         USAGE: ./run.sh 
# 
#   DESCRIPTION: 
# 
#       OPTIONS: ---
#  REQUIREMENTS: ---
#          BUGS: ---
#         NOTES: ---
#        AUTHOR: Dilawar Singh (), dilawars@ncbs.res.in
#  ORGANIZATION: NCBS Bangalore
#       CREATED: 10/06/2016 02:36:25 PM
#      REVISION:  ---
#===============================================================================

set -x
set -o nounset                              # Treat unset variables as an error
cmake .
make 
./jugnoo  ~/Work/OTHERS/Suite2P/g5_136/test/Trial1-ROI-1.tif
