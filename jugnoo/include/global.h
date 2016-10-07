/*
 * =====================================================================================
 *
 *       Filename:  global.h
 *
 *    Description:  Global declarations.
 *
 *        Version:  1.0
 *        Created:  10/07/2016 10:10:03 AM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Dilawar Singh (), dilawars@ncbs.res.in
 *   Organization:  NCBS Bangalore
 *
 * =====================================================================================
 */


#ifndef  global_INC
#define  global_INC

#include "prettyprint.hpp"
#include <utility>
#include <gsl/gsl_matrix.h>
#include "tiffio.h"

using namespace std;

typedef uint16 pixal_type_t;
typedef pair<int, int> index_type_t;
typedef gsl_matrix* matrix_type_t;

#endif   /* ----- #ifndef global_INC  ----- */
