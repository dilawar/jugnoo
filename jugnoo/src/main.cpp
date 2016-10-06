/*
 * =====================================================================================
 *
 *       Filename:  main.cpp
 *
 *    Description:  Entry program.
 *
 *        Version:  1.0
 *        Created:  10/06/2016 02:32:22 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dilawar Singh (), dilawars@ncbs.res.in
 *   Organization:  NCBS Bangalore
 *
 * =====================================================================================
 */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
//#include <boost/git/gil_all.hpp>

using namespace std;
using namespace cv;
//using namespace boost::gil;

int main(int argc, char *argv[])
{
    if( argc != 2 )
    {
        cout << "Usage: ./jugnoo tiff_file" << endl;
        return -1;
    }

#if 1
    Mat image;
    image = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE );   // Read the file

    if(! image.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.

    waitKey(0);                                        
#else


#endif 
    return 0;
}
