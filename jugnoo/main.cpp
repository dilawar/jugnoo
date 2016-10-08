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

#include "global.h"
#include <gsl/gsl_matrix.h>
#include "core.h"

using namespace std;

void test( )
{
    vector< double > a = { 1, 2, 3, 4, 5, 6, 3, 2, 3, 2, 1 };
    vector< double > kernal = { 1.0/3, 1.0/3, 1.0/3 };
    vector< double > sm;
    //smooth( a, sm );
    convolve( a, kernal, sm );
    cout << a << endl;
    cout << kernal << endl;
    cout << sm << endl;
    cout << endl;

}

int main(int argc, char *argv[])
{
    test();

    if( argc != 2 )
    {
        cout << "Usage: ./jugnoo tiff_file" << endl;
        return -1;
    }

    TIFF *tif = TIFFOpen( argv[1], "r");
    std::vector< matrix_type_t > frames;

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

            gsl_matrix* image = gsl_matrix_alloc( h, w );
            gsl_matrix_set_all( image, 0.0 );

            pixal_type_t* row =  new pixal_type_t[ w ];
            for (size_t i = 0; i < h; i++) 
            {
                TIFFReadScanline( tif, row, i, 0 );
                for (size_t ii = 0; ii < w; ii++) 
                    gsl_matrix_set( image, i, ii, row[ii] );
            }
            delete row;
            frames.push_back( image );

	} while (TIFFReadDirectory(tif));
	printf("[INFO] Done reading %d images from %s\n", frames.size(), argv[1]);
    }
    TIFFClose(tif);
    compute_correlation( frames );
    return 0;
}
