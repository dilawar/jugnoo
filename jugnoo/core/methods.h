/*
 * =====================================================================================
 *
 *       Filename:  methods.h
 *
 *    Description:  Methods.
 *
 *        Version:  1.0
 *        Created:  10/07/2016 10:09:27 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dilawar Singh (), dilawars@ncbs.res.in
 *   Organization:  NCBS Bangalore
 *
 * =====================================================================================
 */

#ifndef  methods_INC
#define  methods_INC

#include "global.h"
#include <vector>
#include <valarray>
#include <chrono>
#include <iostream>
#include <map>
#include <numeric>
#include <algorithm>
#include <fstream>

using namespace std;

void get_timeseries_of_pixal( 
        const vector< matrix_type_t > & frames
        , index_type_t index
        , vector< pixal_type_t >& values
        );

/**
 * @brief Convolve two vectors. Results length is max( a.size(), b.size() ).
 * This is equivalent to numpy.convolve with option 'same'.
 *
 * @tparam IN
 * @tparam OUT
 * @param first
 * @param 
 * @param 
 */
template<typename T = double >
void convolve1( const vector< T >& first, const vector< T >& second
        , vector< T >& result 
        )
{
    result.resize( first.size(), 0.0);

#if 0

    /*-----------------------------------------------------------------------------
     *  This solution is slower, 1.3 seconds vs 1.0 second.
     *-----------------------------------------------------------------------------*/
    // At each iteration first vector is indexed from ai to end and second from
    // 0 to bj. There are total a.size() iteration.
    valarray<T> a, b;
    a = valarray<T>( first.data(), first.size() );
    b = valarray<T>( second.data(), second.size() );

    for (size_t i = 0; i < a.size(); i++) 
    {
        valarray<T> sliceA = a[ slice(i, a.size() - i, 1 ) ];
        result[i] = (sliceA * b).sum();
    }

#else

    vector<T> a( first );
    for (size_t i = 0; i < second.size(); i++) 
        a.push_back( 0.0 );
        
    for (size_t i = 0; i < first.size(); i++) 
    {
        // NOTE: slice is costly.
        //auto sliceA = valarray<T>( &a[i], a.size() - i );
        T sum = 0.0;
        for( size_t ii = 0; ii < second.size(); ii++ )
            sum += a[i+ii] * second[ii];
        result[i] = sum;
    }

#endif 
}

template<typename T = double >
void convolve( const vector< T >& first, const vector< T >& second
        , vector< T >& result 
        )
{
    // Fisrt vector must be bigger or equal to the second one.
    if( first.size() >= second.size() )
        convolve1( first, second, result );
    else
        convolve1( second, first, result );
}

/**
 * @brief Smooth out a singal by a window of size N
 *
 * @tparam T
 * @param signal
 * @param N
 * @param res
 */
template< typename T = double >
void smooth( const vector< T >& signal, const size_t N, vector < T >& res ) 
{
    vector< double > kernal(N);
    for (size_t i = 0; i < N; i++) 
        kernal[i] = 1.0 /N ;
    convolve( signal, kernal, res );
}


template< typename T = double >
double correlate( const vector<T>& first, const vector<T>& second)
{
    // Normalize both a and b first.
    double maxA = * max_element( first.begin(), first.end() );
    double maxB = * max_element( second.begin(), second.end() );

    vector<T> a, b;
    for_each( first.begin(), first.end(), [&](T v) { a.push_back( v / maxA ); });
    for_each( second.begin(), second.end(), [&](T v) { b.push_back( v / maxB ); });

    double err = 0.0;
    for (size_t i = 0; i < a.size(); i++) 
        err += fabs( a[i] - b[i] ) / a[i];

    return err / first.size( );
}

template<typename T = double >
bool is_active_timeseries( const vector<T>& data )
{
    //cout << data << endl;
    // If mean and std deviation is smaller then there is not much actvity here.
    T min = 100000.0, max = 0.0, sum = 0.0, mean = 0.0;
    vector<T> normalized( data.size(), 0.0 );
    for( auto v : data )
    {
        sum += v;
        if( v < min )
            min = v;
        if( v > max )
            max = v;
    }

    mean = sum / data.size( );
    double accum = 0.0;
    std::for_each ( std::begin(data), std::end(data)
            , [mean,&accum](const double d) { accum += (d-mean) * (d-mean); }
            );
    double stdev = sqrt( accum / (data.size()-1));
    double cv = stdev / mean;
#if 0
    cout << stdev << " " << mean;
    cout << "cv: " << cv << endl;
#endif
    if( cv > 0.15 )
        return true;
    return false;
}

template<typename T = double >
void get_timeseries_of_pixal( 
        const vector< matrix_type_t > & frames
        , index_type_t index
        , vector< T >& values
        )
{
    for ( matrix_type_t image : frames )
    {
        auto val = gsl_matrix_get( image, index.first, index.second );
        values.push_back( val );
    }
}


void compute_correlation( const vector< matrix_type_t >& frames )
{
    auto frame0 = frames[0];
    size_t rows = frame0->size1;
    size_t cols = frame0->size2;

    std::cout << "Rows " << rows << " cols : " << cols << std::endl;

    vector< vector< double > > time_series;

    // create the list of indices, I need to iterate over. And also collect the
    // time-series of pixals.
    vector< pair<size_t, size_t> > indices;
    map< tuple<size_t, size_t>, vector<double> > pixalMap;
    for( int i = 0; i < rows; i++)
        for( int ii = 0; ii < cols; ii++ )
        {
            auto index = make_pair( i, ii );
            vector<double> pixal;
            get_timeseries_of_pixal( frames, index, pixal );
            if( is_active_timeseries( pixal ) )
            {
                indices.push_back( index );
                pixalMap[index] = pixal;
            }
        }

    std::cout << "Total pixals to process " << pixalMap.size() << endl;
    long count = 0;

    std::chrono::time_point< std::chrono::system_clock> start, t;
    std::chrono::duration<double> duration;

    ofstream file;
    string datafile("data_correlation.dot");
    file.open( datafile );
    file << "digraph corr { " << endl;
    file.close();

    start = std::chrono::system_clock::now();
    for (size_t i = 0; i < pixalMap.size(); i++) 
    {
        t = std::chrono::system_clock::now();
        duration = t - start;
        printf( "| %.3f%% (out of %d) completed. Elapsed time %f \n"
                , (100.0 * i) / pixalMap.size(), pixalMap.size(), duration.count() );
        for (size_t j = i; j < pixalMap.size(); j++) 
        {
            count++;
            vector<double> a, b;
            smooth( pixalMap[ indices[i] ], 3, a);
            smooth( pixalMap[ indices[j] ], 3, b);
            double corr = correlate( a, b );

            // Write it to file.
            size_t row1, row2, col1, col2;
            row1 = indices[i].first; col1 = indices[i].second;
            row2 = indices[j].first; col2 = indices[j].second;
            file.open( datafile, ios::app );
            file << "\t \"(" << row1 << "," << col1 << ")\" -> " 
                << "\"(" << row2 << "," << col2  << ")\" " 
                <<  "[weight=" << corr << "];" << endl; 
            file.close( );
        }
    }

    file.open( datafile, ios::app );
    file << "}" << endl;
    file.close( );

    cout << "Wrote data to " << datafile << endl;
}
#endif   /* ----- #ifndef methods_INC  ----- */
