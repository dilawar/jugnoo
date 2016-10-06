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

#include <iostream>
#include <vector>
#include <array>
#include <tuple>
#include <valarray>
#include <numeric>
#include <opencv2/opencv.hpp>
#include "tiffio.h"

using namespace std;
using namespace cv;

typedef unsigned short pixal_type_t;
typedef tuple<int, int> index_type_t;

//typedef adjacency_list<vecS, vecS, bidirectionalS> Graph;

void get_timeseries_of_pixal( 
        const vector< Mat > & frames
        , index_type_t index
        , vector< pixal_type_t >& values
        )
{
    for ( auto image : frames )
    {

        pixal_type_t val = image.at<pixal_type_t>( get<0>(index), get<1>(index) );
        values.push_back( val );
    }
}

void smooth( const vector< pixal_type_t > signal
        , vector < pixal_type_t >& res 
        , size_t window_size 
        )
{


}

void convolve( const vector< pixal_type_t > a, const vector< pixal_type_t > b
        , vector< pixal_type_t >& result 
        )
{
    for (size_t i = 0; i < a.size(); i++) 
    {
        valarray<pixal_type_t> a1( &a[i], a.size() );
        valarray<pixal_type_t> b1( &b[0], a1.size() ); 
        result.push_back( (a1*b1).sum() );
    }
}

void correlate( const vector< pixal_type_t> a, const vector<pixal_type_t> b)
{
}

void compute_correlation( const vector< Mat >& frames )
{
    auto frame0 = frames[0];
    size_t rows, cols;
    cols = frame0.cols; rows = frame0.rows;

    std::cout << "Rows " << rows << " cols : " << cols << std::endl;

    vector< vector< pixal_type_t > > time_series;

    // create the list of indices, I need to iterate over.
    vector< tuple<size_t, size_t> > indices;
    for( int i = 0; i < rows; i++)
        for( int ii = 0; i < cols; i++ )
            indices.push_back( make_tuple( i, ii ) );

    int count = 0;
    for( auto i : indices )
        for( auto j : indices )
        {
            if( i <= j )
                continue;
            count++;
            vector<pixal_type_t> pixalA, pixalB;
            get_timeseries_of_pixal( frames, i, pixalA);
            get_timeseries_of_pixal( frames, j, pixalB);
            vector< pixal_type_t > convolution;
            convolve( pixalA, pixalB, convolution);
            //cout << "," << accumulate( convolution.begin(), convolution.end(), 0);
        }
    cout << "Total pixals " << count << endl;
}

int main(int argc, char *argv[])
{
    if( argc != 2 )
    {
        cout << "Usage: ./jugnoo tiff_file" << endl;
        return -1;
    }

    TIFF *tif = TIFFOpen( argv[1], "r");
    std::vector< Mat > frames;

    if (tif) {
	int dircount = 0;
        
        // Iterate over each frame now.
	do 
        {
	    dircount++;
            uint32 w, h;
            size_t npixals;

            TIFFGetField( tif, TIFFTAG_IMAGEWIDTH, &w);
            TIFFGetField( tif, TIFFTAG_IMAGELENGTH, &h);
            npixals = w * h;

            Mat image( h, w, CV_8UC1 );
            for (size_t i = 0; i < h; i++) 
            {
                pixal_type_t* row =  new pixal_type_t[ w ];
                TIFFReadScanline( tif, row, i, 0 );
                for (size_t ii = 0; ii < w; ii++) 
                    image.at<uchar>(i, ii) = row[ii];
                delete row;
            }

            frames.push_back( image );

	} while (TIFFReadDirectory(tif));
	printf("[INFO] Done reading %d images from %s\n", frames.size(), argv[1]);
    }
    TIFFClose(tif);
    compute_correlation( frames );
    return 0;
}
