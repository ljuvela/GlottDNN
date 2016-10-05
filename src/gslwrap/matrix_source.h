// matrix.h

//  This matrix class is a C++ wrapper for the GNU Scientific Library
//  Copyright (C) 2001 Ramin Nakisa

//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.

//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.

//  You should have received a copy of the GNU General Public License
//  along with this program; if not, write to the Free Software
//  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

#if !defined( _matrix_#type#_h )
#define _matrix_#type#_h

#ifdef __HP_aCC //for aCC B3910B A.01.27
#include <iostream.h>
#include <fstream.h>
#include <iomanip.h>
#else //for gcc3
#include <iostream>
#include <fstream>
#include <iomanip>
#endif

#include <math.h>
#include <stdlib.h>
#include <assert.h>
///
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix#typeext#.h>
#include <gsl/gsl_linalg.h>
#include <gslwrap/permutation.h>
#include <gslwrap/vector_#type#.h>

#define type_is#typeext#
#ifdef  type_is
#define type_is_double
#endif

namespace gsl
{

///
class matrix#typeext#
{
#ifdef type_is_double
	friend class matrix_float;
	friend class matrix_int;
#endif
public:
	typedef #type# value_type;
	typedef vector#typeext# vector_type;

	///
	matrix#typeext#();
	///
	matrix#typeext#( size_t new_rows, size_t new_cols, bool clear = true );

	template<class oclass>
	void copy(const oclass &other)
	{
		if ( static_cast<const void *>( this ) == static_cast<const void *>( &other ) )
			return;

		set_dimensions( other.get_rows(), other.get_cols() );
		for ( size_t i = 0; i < get_rows(); i++ ) 
		{
			for ( size_t j = 0; j < get_cols(); j++ ) 
			{
				gsl_matrix#typeext#_set( m, i, j, (#type#)other(i,j));
			}
		}
	}
/*    	template<>  */
/*    	void copy<matrix#typeext#>(const matrix#typeext# &other)  */
/*    	{  */
/*    		set_dimensions(other.size1(),other.size2());  */
/*    		gsl_matrix#typeext#_memcpy( m, other.m );  */
/*    	}  */
	// copy constructor for type matrix#typeext#
	matrix#typeext#( const matrix#typeext# &other ):m(NULL) {copy(other);}
	///
	template<class oclass>
	matrix#typeext#( const oclass &other ):m(NULL) {copy(other);}

	///
	~matrix#typeext#();
	///
//	matrix#typeext#( const char *Filename );
	///
	size_t get_rows() const {return m->size1;}
	///
	size_t get_cols() const {return m->size2;}
	///
	size_t size1() const {return m->size1;}
	///
	size_t size2() const {return m->size2;}
   

	///
	void dimensions( size_t *num_rows, size_t *num_cols ) const;
	///
#type#        get_element ( size_t row, size_t col ) const {return  gsl_matrix#typeext#_get( m, row, col ) ;}
	const #type# &operator()( size_t row, size_t col ) const {return *gsl_matrix#typeext#_ptr( m, row, col ) ;}
#type#       &operator()( size_t row, size_t col )       {return *gsl_matrix#typeext#_ptr( m, row, col ) ;}
	///
	void set_element( size_t row, size_t col, const #type# &v ){ gsl_matrix#typeext#_set( m, row, col, v );}
	///
	void set_elements( const #type# & new_value );
	void set_all ( const #type# & new_value ) {gsl_matrix#typeext#_set_all ( m, new_value );}
	void set_zero() {gsl_matrix#typeext#_set_zero( m );}
	///
	void set_dimensions( size_t new_rows, size_t new_cols );
	///
	void load( const char *filename );
	///
	void save( const char *filename ) const;
	///
	friend ostream& operator<< ( ostream& os, const matrix#typeext#& m );
	//This function writes the elements of the matrix m to the stream stream in binary format. The return value is 0 for success and GSL_EFAILED if there was a problem writing to the file. Since the data is written in the native binary format it may not be portable between different architectures.
	int fwrite (FILE * stream) const {return gsl_matrix#typeext#_fwrite (stream, m);}

//This function reads into the matrix m from the open stream stream in binary format. The matrix m must be preallocated with the correct dimensions since the function uses the size of m to determine how many bytes to read. The return value is 0 for success and GSL_EFAILED if there was a problem reading from the file. The data is assumed to have been written in the native binary format on the same architecture. 
	int fread (FILE * stream) {return gsl_matrix#typeext#_fread (stream, m);}

    ///
	void load_binary( const char *filename );
	///
	void save_binary( const char *filename ) const;
	///
	bool operator==( const matrix#typeext# &other ) const;
	bool operator!=( const matrix#typeext# &other ) const {return !((*this)==other);}
	
	matrix#typeext#& operator=( const matrix#typeext# &other ) {copy( other );return *this;}
	/// converts from any other matrix type
	template<class omatrix>
	matrix#typeext# &operator=( const omatrix& other )
	{
			copy(other);
			return *this;
	}
   ///
	matrix#typeext# operator+( const matrix#typeext# &other ) const;
	///
	matrix#typeext# operator+( const #type# &f ) const;
	///
	friend matrix#typeext# operator+( const #type# &f, const matrix#typeext# &other );
	///
	matrix#typeext# &operator+=( const #type# &f );
	///
	matrix#typeext# &operator+=( const matrix#typeext# &other );
	///
	matrix#typeext# operator-( const matrix#typeext# &other ) const;
	///
	matrix#typeext# operator-( const #type# &f ) const;
	///
	friend matrix#typeext# operator-( const #type# &f, const matrix#typeext# &other );
	///
	matrix#typeext# &operator-=( const #type# &f );
	///
	matrix#typeext# &operator-=( const matrix#typeext# &other );
	///
	matrix#typeext# operator*( const matrix#typeext# &other ) const;
	///
	matrix#typeext# operator*( const #type# &f ) const;
	///
	friend matrix#typeext# operator*( const #type# &f, const matrix#typeext# &other );
	///
	matrix#typeext# &operator*=( const #type# &f );
	///
	matrix#typeext# &operator*=( const matrix#typeext# &other );
	///
	matrix#typeext# operator/( const #type# &) const;
	///
	matrix#typeext# &operator/=( const #type# &);
	///
	matrix#typeext# transpose() const;
	///
	matrix#typeext# LU_decomp(gsl::permutation *perm=NULL,int *psign=NULL) const;
	///
	matrix#typeext# LU_invert() const;

	// return a submatrix of the this from row_min to row_max (not included!)
	matrix#typeext# submatrix(size_t row_min, size_t row_max, size_t col_min, size_t col_max) const 
		{
			matrix#typeext# m(row_max - row_min, col_max - col_min);
			for (size_t i = row_min ; i < row_max ; i++)
			{
				for (size_t j = col_min ; j < col_max ; j++)
				{
					m(i - row_min,j - col_min) = (*this)(i,j);
				}
			}
			return m;
		}
private:
	///
	void LU_decomp( gsl_matrix#typeext# **a,
					gsl_permutation **permutation,
					int *sign ) const;
public:
	/** returns sum of all the matrix elements. */
    #type# sum() const;
	/** returns logarithm of the determinant of the matrix. */
	double LU_lndet() const;


	/** returns a vector#typeext#_view of a single row of the matrix. */
	vector#typeext#_view       row( size_t rowindex );
	const vector#typeext#_view row( size_t rowindex ) const ;
	/** returns a vector#typeext#_view of a single column of the matrix. */
	vector#typeext#_view       column( size_t colindex );
	const vector#typeext#_view column( size_t colindex ) const;
	/** returns a vector#typeext#_view of the diagonal elements of the matrix. */
	vector#typeext#_view       diagonal();
	const vector#typeext#_view diagonal() const;

	/** returns a column matrix containing a single row of the matrix. */
	matrix#typeext# get_row( size_t rowindex ) const;
	/** returns a column matrix containing a single column of the matrix. */
	matrix#typeext# get_col( size_t colindex ) const;
	/** calculates sum of rows returned as a column matrix. */
	matrix#typeext# row_sum() const;
	/** calculates sum of columns returned as a row matrix. */
	matrix#typeext# column_sum() const;
	/** returns trace (diagonal sum) of a square matrix. */
	double trace() const;
	/** calculates cholesky decomposition of the matrix, returning success if matrix is positive definite. */
	int cholesky_decomp( matrix#typeext# &a ) const;
//  	/** returns index of nearest row in matrix to vector argument. */
//  	int nearest_row_index( const matrix#typeext# &v ) const;
	/** calculates covariance of the matrix columns. */
	matrix#typeext# covariance() const;
	/** returns 1 if matrix is square, 0 otherwise. */
	bool is_square() const;
	/** diag operator (sets the diagonal elements of the matrix to the elements of v */
	void diag(const vector#typeext#& v);
	/** set diagonal elements of a square matrix to f. */
	void set_diagonal( #type# f );
	/** sets matrix to a k dimensional unit matrix. */
	void identity( size_t k );
	/** returns sum of nth power of all elements. */
	double norm( double n ) const;

/*  Function: double gsl_matrix_max (const gsl_matrix * m)  */
/*      This function returns the maximum value in the matrix m.  */
	double max() const {return gsl_matrix#typeext#_max(m);}
/*  Function: double gsl_matrix_min (const gsl_matrix * m)  */
/*      This function returns the minimum value in the matrix m.  */
	double min()const{return gsl_matrix#typeext#_min(m);}

	/** This function returns 1 if all the elements of the matrix m are zero, and 0 otherwise. */
	bool isnull() const { return gsl_matrix#typeext#_isnull(m);}
/*  Function: void gsl_matrix_minmax (const gsl_matrix * m, double * min_out, double * max_out)  */
/*      This function returns the minimum and maximum values in the matrix m, storing them in min_out and max_out.  */

/*  Function: void gsl_matrix_max_index (const gsl_matrix * m, size_t * imax, size_t * jmax)  */
/*      This function returns the indices of the maximum value in the matrix m, storing them in imax and jmax. When there are several equal maximum elements then the first element found */
/*      is returned.  */

/*  Function: void gsl_matrix_min_index (const gsl_matrix * m, size_t * imax, size_t * jmax)  */
/*      This function returns the indices of the minimum value in the matrix m, storing them in imax and jmax. When there are several equal minimum elements then the first element found */
/*      is returned.  */

/*  Function: void gsl_matrix_minmax_index (const gsl_matrix * m, size_t * imin, size_t * imax)  */
/*      This function returns the indices of the minimum and maximum values in the matrix m, storing them in (imin,jmin) and (imax,jmax). When there are several equal minimum or */
/*      maximum elements then the first elements found are returned.  */

	/** for interfacing with gsl c */
/*  	gsl_matrix#typeext#       *gslobj()       {if (!m){cout << "matrix#typeext#::gslobj ERROR, data not initialized!! " << endl; exit(-1);}return m;} */
/*  	const gsl_matrix#typeext# *gslobj() const {if (!m){cout << "matrix#typeext#::gslobj ERROR, data not initialized!! " << endl; exit(-1);}return m;} */
	gsl_matrix#typeext#       *gslobj()       {assert(m);return m;}
	const gsl_matrix#typeext# *gslobj() const {assert(m);return m;}
private:
	///
   gsl_matrix#typeext# *m;

};
}
#undef type_is#typeext#
#undef type_is_double

#endif // _matrix_#type#_h
