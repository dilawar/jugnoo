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

#include <tiffio.h>

int main(int argc, char *argv[])
{
    TIFF* tif = TIFFOpen( argv[1], "r" );

    TIFFClose( tif );
    return 0;
}
