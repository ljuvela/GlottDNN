//  This matrix class is a C++ wrapper for the GNU Scientific Library
//  Copyright (C)  ULP-IPB Strasbourg

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

#ifndef _vector_#type#_h
#define _vector_#type#_h

#ifdef __HP_aCC
#include <iostream.h>
#else 
#include <iostream>
#endif

#include <gsl/gsl_math.h>
#include <gsl/gsl_vector#typeext#.h>
#include <gsl/gsl_blas.h>
#include <gslwrap/vector_double.h>

//#define NDEBUG 0

#include <assert.h>
namespace gsl
{

#ifndef __HP_aCC
	using std::ostream;
//using std::string;
//using std::runtime_error;
#endif

class vector#typeext#_view;

class vector#typeext#
{
protected:
	gsl_vector#typeext# *gsldata;
	void free(){if(gsldata) gsl_vector#typeext#_free(gsldata);gsldata=NULL;}
	void alloc(size_t n) {gsldata=gsl_vector#typeext#_alloc(n);}
	void calloc(size_t n){gsldata=gsl_vector#typeext#_calloc(n);}
public:
	typedef #type# value_type;
	vector#typeext#() : gsldata(NULL) {;}
	vector#typeext#( const vector#typeext# &other ):gsldata(NULL) {copy(other);}
	template<class oclass>
	vector#typeext#( const oclass &other ):gsldata(NULL) {copy(other);}
	~vector#typeext#(){free();}
	vector#typeext#(const size_t& n,bool clear=true)
	{
		if(clear){this->calloc(n);}
		else     {this->alloc(n);}
	}
	vector#typeext#(const int& n,bool clear=true)
	{
		if(clear){this->calloc(n);}
		else     {this->alloc(n);}
	}
	
	void resize(size_t n);

	template <class oclass>
		void copy(const oclass &other)
		{
			if ( static_cast<const void *>( this ) == static_cast<const void *>( &other ) )
				return;

			if (!other.is_set())
			{
				gsldata=NULL;
				return;
			}
			resize(other.size());
			for (size_t i=0;i<size();i++)
			{
				gsl_vector#typeext#_set(gsldata, i, (#type#)other[i]);
			}
		}
	void copy(const vector#typeext#& other);
	bool is_set() const{if (gsldata) return true; else return false;}
//	void clone(vector#typeext#& other);
	
//	size_t size() const {if (!gsldata) {cout << "vector#typeext#::size vector not initialized" << endl; exit(-1);}return gsldata->size;}
	size_t size() const {assert (gsldata); return gsldata->size;}

	/** for interfacing with gsl c */
/*  	gsl_vector#typeext#       *gslobj()       {if (!gsldata){cout << "vector#typeext#::gslobj ERROR, data not initialized!! " << endl; exit(-1);}return gsldata;} */
/*  	const gsl_vector#typeext# *gslobj() const {if (!gsldata){cout << "vector#typeext#::gslobj ERROR, data not initialized!! " << endl; exit(-1);}return gsldata;} */
	gsl_vector#typeext#       *gslobj()       {assert(gsldata);return gsldata;}
	const gsl_vector#typeext# *gslobj() const {assert(gsldata);return gsldata;}


	static vector#typeext#_view create_vector_view( const gsl_vector#typeext#_view &other );

// ********Accessing vector elements

//  Unlike FORTRAN compilers, C compilers do not usually provide support for range checking of vectors and matrices (2). However, the functions gsl_vector#typeext#_get and gsl_vector#typeext#_set can perform range checking for you and report an error if you attempt to access elements outside the allowed range. 

//  The functions for accessing the elements of a vector or matrix are defined in `gsl_vector#typeext#.h' and declared extern inline to eliminate function-call overhead. If necessary you can turn off range checking completely without modifying any source files by recompiling your program with the preprocessor definition GSL_RANGE_CHECK_OFF. Provided your compiler supports inline functions the effect of turning off range checking is to replace calls to gsl_vector#typeext#_get(v,i) by v->data[i*v->stride] and and calls to gsl_vector#typeext#_set(v,i,x) by v->data[i*v->stride]=x. Thus there should be no performance penalty for using the range checking functions when range checking is turned off. 

//      This function returns the i-th element of a vector v. If i lies outside the allowed range of 0 to n-1 then the error handler is invoked and 0 is returned. 
	#type# get(size_t i) const {return gsl_vector#typeext#_get(gsldata,i);}

//      This function sets the value of the i-th element of a vector v to x. If i lies outside the allowed range of 0 to n-1 then the error handler is invoked. 
	void  set(size_t i,#type# x){gsl_vector#typeext#_set(gsldata,i,x);}

//      These functions return a pointer to the i-th element of a vector v. If i lies outside the allowed range of 0 to n-1 then the error handler is invoked
	#type#       &operator[](size_t i)       { return *gsl_vector#typeext#_ptr(gsldata,i);}
	const #type# &operator[](size_t i) const { return *gsl_vector#typeext#_ptr(gsldata,i);}

	#type#       &operator()(size_t i)       { return *gsl_vector#typeext#_ptr(gsldata,i);}
	const #type# &operator()(size_t i) const { return *gsl_vector#typeext#_ptr(gsldata,i);}


//  ***** Initializing vector elements

//      This function sets all the elements of the vector v to the value x. 
	void set_all(#type# x){gsl_vector#typeext#_set_all (gsldata,x);}
//      This function sets all the elements of the vector v to zero. 
	void set_zero(){gsl_vector#typeext#_set_zero (gsldata);}

//      This function makes a basis vector by setting all the elements of the vector v to zero except for the i-th element which is set to one. 
	int set_basis (size_t i) {return gsl_vector#typeext#_set_basis (gsldata,i);}

//  **** Reading and writing vectors

//  The library provides functions for reading and writing vectors to a file as binary data or formatted text. 


//      This function writes the elements of the vector v to the stream stream in binary format. The return value is 0 for success and GSL_EFAILED if there was a problem writing to the file. Since the data is written in the native binary format it may not be portable between different architectures. 
	int fwrite (FILE * stream) const {return gsl_vector#typeext#_fwrite (stream, gsldata);}

//      This function reads into the vector v from the open stream stream in binary format. The vector v must be preallocated with the correct length since the function uses the size of v to determine how many bytes to read. The return value is 0 for success and GSL_EFAILED if there was a problem reading from the file. The data is assumed to have been written in the native binary format on the same architecture. 
	int fread (FILE * stream) {return gsl_vector#typeext#_fread (stream, gsldata);}

	void load( const char *filename );
	///
	void save( const char *filename ) const;

//      This function writes the elements of the vector v line-by-line to the stream stream using the format specifier format, which should be one of the %g, %e or %f formats for floating point numbers and %d for integers. The function returns 0 for success and GSL_EFAILED if there was a problem writing to the file. 
	int fprintf (FILE * stream, const char * format) const {return gsl_vector#typeext#_fprintf (stream, gsldata,format) ;}

//      This function reads formatted data from the stream stream into the vector v. The vector v must be preallocated with the correct length since the function uses the size of v to determine how many numbers to read. The function returns 0 for success and GSL_EFAILED if there was a problem reading from the file. 
	int fscanf (FILE * stream)  {return gsl_vector#typeext#_fscanf (stream, gsldata); }




//  ******* Vector views

//  In addition to creating vectors from slices of blocks it is also possible to slice vectors and create vector views. For example, a subvector of another vector can be described with a view, or two views can be made which provide access to the even and odd elements of a vector. 

//  A vector view is a temporary object, stored on the stack, which can be used to operate on a subset of vector elements. Vector views can be defined for both constant and non-constant vectors, using separate types that preserve constness. A vector view has the type gsl_vector#typeext#_view and a constant vector view has the type gsl_vector#typeext#_const_view. In both cases the elements of the view can be accessed as a gsl_vector#typeext# using the vector component of the view object. A pointer to a vector of type gsl_vector#typeext# * or const gsl_vector#typeext# * can be obtained by taking the address of this component with the & operator. 

//      These functions return a vector view of a subvector of another vector v. The start of the new vector is offset by offset elements from the start of the original
//      vector. The new vector has n elements. Mathematically, the i-th element of the new vector v' is given by, 

//      v'(i) = v->data[(offset + i)*v->stride]

//      where the index i runs from 0 to n-1. 

//      The data pointer of the returned vector struct is set to null if the combined parameters (offset,n) overrun the end of the original vector. 

//      The new vector is only a view of the block underlying the original vector, v. The block containing the elements of v is not owned by the new vector. When the
//      new vector goes out of scope the original vector v and its block will continue to exist. The original memory can only be deallocated by freeing the original vector.
//      Of course, the original vector should not be deallocated while the new vector is still in use. 

//      The function gsl_vector#typeext#_const_subvector is equivalent to gsl_vector#typeext#_subvector but can be used for vectors which are declared const. 
	vector#typeext#_view subvector (size_t offset, size_t n);
	const vector#typeext#_view subvector (size_t offset, size_t n) const;
//	vector#typeext#_const_view subvector (size_t offset, size_t n) const;

//  	class view
//  	{
//  		gsl_vector#typeext#_view *gsldata;
//  	public:
//  		view();
//  	};
//  	view subvector(size_t offset, size_t n)
//  	{
//  		return view(gsl_vector#typeext#_subvector(gsldata,offset,n);
//  	}
//  	const view subvector(size_t offset, size_t n) const
//  	{
//  		return view(gsl_vector#typeext#_const_subvector(gsldata,offset,n);
//  	}



//  Function: gsl_vector#typeext# gsl_vector#typeext#_subvector_with_stride (gsl_vector#typeext# *v, size_t offset, size_t stride, size_t n) 
//  Function: gsl_vector#typeext#_const_view gsl_vector#typeext#_const_subvector_with_stride (const gsl_vector#typeext# * v, size_t offset, size_t stride, size_t n) 
//      These functions return a vector view of a subvector of another vector v with an additional stride argument. The subvector is formed in the same way as for
//      gsl_vector#typeext#_subvector but the new vector has n elements with a step-size of stride from one element to the next in the original vector. Mathematically,
//      the i-th element of the new vector v' is given by, 

//      v'(i) = v->data[(offset + i*stride)*v->stride]

//      where the index i runs from 0 to n-1. 

//      Note that subvector views give direct access to the underlying elements of the original vector. For example, the following code will zero the even elements of the
//      vector v of length n, while leaving the odd elements untouched, 

//      gsl_vector#typeext#_view v_even = gsl_vector#typeext#_subvector_with_stride (v, 0, 2, n/2);
//      gsl_vector#typeext#_set_zero (&v_even.vector);

//      A vector view can be passed to any subroutine which takes a vector argument just as a directly allocated vector would be, using &view.vector. For example, the
//      following code computes the norm of odd elements of v using the BLAS routine DNRM2, 

//      gsl_vector#typeext#_view v_odd = gsl_vector#typeext#_subvector_with_stride (v, 1, 2, n/2);
//      double r = gsl_blas_dnrm2 (&v_odd.vector);

//      The function gsl_vector#typeext#_const_subvector_with_stride is equivalent to gsl_vector#typeext#_subvector_with_stride but can be used for
//      vectors which are declared const. 

//  Function: gsl_vector#typeext#_view gsl_vector#typeext#_complex_real (gsl_vector#typeext#_complex *v) 
//  Function: gsl_vector#typeext#_const_view gsl_vector#typeext#_complex_const_real (const gsl_vector#typeext#_complex *v) 
//      These functions return a vector view of the real parts of the complex vector v. 

//      The function gsl_vector#typeext#_complex_const_real is equivalent to gsl_vector#typeext#_complex_real but can be used for vectors which are declared
//      const. 

//  Function: gsl_vector#typeext#_view gsl_vector#typeext#_complex_imag (gsl_vector#typeext#_complex *v) 
//  Function: gsl_vector#typeext#_const_view gsl_vector#typeext#_complex_const_imag (const gsl_vector#typeext#_complex *v) 
//      These functions return a vector view of the imaginary parts of the complex vector v. 

//      The function gsl_vector#typeext#_complex_const_imag is equivalent to gsl_vector#typeext#_complex_imag but can be used for vectors which are declared
//      const. 

//  Function: gsl_vector#typeext#_view gsl_vector#typeext#_view_array (double *base, size_t n) 
//  Function: gsl_vector#typeext#_const_view gsl_vector#typeext#_const_view_array (const double *base, size_t n) 
//      These functions return a vector view of an array. The start of the new vector is given by base and has n elements. Mathematically, the i-th element of the new
//      vector v' is given by, 

//      v'(i) = base[i]

//      where the index i runs from 0 to n-1. 

//      The array containing the elements of v is not owned by the new vector view. When the view goes out of scope the original array will continue to exist. The
//      original memory can only be deallocated by freeing the original pointer base. Of course, the original array should not be deallocated while the view is still in use. 

//      The function gsl_vector#typeext#_const_view_array is equivalent to gsl_vector#typeext#_view_array but can be used for vectors which are declared const. 

//  Function: gsl_vector#typeext#_view gsl_vector#typeext#_view_array_with_stride (double * base, size_t stride, size_t n) 
//  Function: gsl_vector#typeext#_const_view gsl_vector#typeext#_const_view_array_with_stride (const double * base, size_t stride, size_t n) 
//      These functions return a vector view of an array base with an additional stride argument. The subvector is formed in the same way as for
//      gsl_vector#typeext#_view_array but the new vector has n elements with a step-size of stride from one element to the next in the original array. Mathematically,
//      the i-th element of the new vector v' is given by, 

//      v'(i) = base[i*stride]

//      where the index i runs from 0 to n-1. 

//      Note that the view gives direct access to the underlying elements of the original array. A vector view can be passed to any subroutine which takes a vector
//      argument just as a directly allocated vector would be, using &view.vector. 

//      The function gsl_vector#typeext#_const_view_array_with_stride is equivalent to gsl_vector#typeext#_view_array_with_stride but can be used for
//      arrays which are declared const. 


//  ************* Copying vectors

//  Common operations on vectors such as addition and multiplication are available in the BLAS part of the library (see section BLAS Support). However, it is useful to have a small number of utility functions which do not require the full BLAS code. The following functions fall into this category. 

//      This function copies the elements of the vector src into the vector dest.
	vector#typeext#& operator=(const vector#typeext#& other){copy(other);return (*this);}

//  Function: int gsl_vector#typeext#_swap (gsl_vector#typeext# * v, gsl_vector#typeext# * w) 
//      This function exchanges the elements of the vectors v and w by copying. The two vectors must have the same length. 

//  ***** Exchanging elements

//  The following function can be used to exchange, or permute, the elements of a vector. 

//  Function: int gsl_vector#typeext#_swap_elements (gsl_vector#typeext# * v, size_t i, size_t j) 
//      This function exchanges the i-th and j-th elements of the vector v in-place. 
	int swap_elements (size_t i, size_t j) {return gsl_vector#typeext#_swap_elements (gsldata, i,j);}

//  Function: int gsl_vector#typeext#_reverse (gsl_vector#typeext# * v) 
//      This function reverses the order of the elements of the vector v. 
	int reverse () {return  gsl_vector#typeext#_reverse (gsldata) ;}

// ******* Vector operations

//  The following operations are only defined for real vectors. 

//      This function adds the elements of vector b to the elements of vector a, a'_i = a_i + b_i. The two vectors must have the same length. 
	int operator+=(const vector#typeext# &other) {return gsl_vector#typeext#_add (gsldata, other.gsldata);}

//      This function subtracts the elements of vector b from the elements of vector a, a'_i = a_i - b_i. The two vectors must have the same length. 
	int operator-=(const vector#typeext# &other) {return gsl_vector#typeext#_sub (gsldata, other.gsldata);}

//  Function: int gsl_vector#typeext#_mul (gsl_vector#typeext# * a, const gsl_vector#typeext# * b) 
//      This function multiplies the elements of vector a by the elements of vector b, a'_i = a_i * b_i. The two vectors must have the same length. 
	int operator*=(const vector#typeext# &other) {return gsl_vector#typeext#_mul (gsldata, other.gsldata);}

//      This function divides the elements of vector a by the elements of vector b, a'_i = a_i / b_i. The two vectors must have the same length. 
	int operator/=(const vector#typeext# &other) {return gsl_vector#typeext#_div (gsldata, other.gsldata);}

//      This function multiplies the elements of vector a by the constant factor x, a'_i = x a_i. 
	int operator*=(#type# x) {return gsl_vector#typeext#_scale (gsldata, x);}

//  Function: int gsl_vector#typeext#_add_constant (gsl_vector#typeext# * a, const double x) 
//      This function adds the constant value x to the elements of the vector a, a'_i = a_i + x. 
	int operator+=(#type# x) {return gsl_vector#typeext#_add_constant (gsldata,x);}

//      This function multiplies the elements of vector a by the constant factor x, a'_i = x a_i. 
	int operator/=(#type# x) {return gsl_vector#typeext#_scale (gsldata, 1/x);}

// bool operators:
	bool operator==(const vector#typeext#& other) const;
	bool operator!=(const vector#typeext#& other) const { return (!((*this)==other));}

// stream output:
//	friend ostream& operator<< ( ostream& os, const vector#typeext#& vect );
	/** returns sum of all the vector elements. */
    #type# sum() const;
	// returns sqrt(v.t*v);
    double norm2() const;


// **** Finding maximum and minimum elements of vectors

//      This function returns the maximum value in the vector v. 
    double max() const{return gsl_vector#typeext#_max (gsldata) ;}

//  Function: double gsl_vector#typeext#_min (const gsl_vector#typeext# * v) 
//      This function returns the minimum value in the vector v. 
    double min() const{return gsl_vector#typeext#_min (gsldata) ;}

//  Function: void gsl_vector#typeext#_minmax (const gsl_vector#typeext# * v, double * min_out, double * max_out) 
//      This function returns the minimum and maximum values in the vector v, storing them in min_out and max_out. 

//      This function returns the index of the maximum value in the vector v. When there are several equal maximum elements then the lowest index is returned. 
	size_t max_index(){return gsl_vector#typeext#_max_index (gsldata);}

//  Function: size_t gsl_vector#typeext#_min_index (const gsl_vector#typeext# * v) 
//      This function returns the index of the minimum value in the vector v. When there are several equal minimum elements then the lowest index is returned. 
	size_t min_index(){return gsl_vector#typeext#_min_index (gsldata);}

//  Function: void gsl_vector#typeext#_minmax_index (const gsl_vector#typeext# * v, size_t * imin, size_t * imax) 
//      This function returns the indices of the minimum and maximum values in the vector v, storing them in imin and imax. When there are several equal minimum
//      or maximum elements then the lowest indices are returned. 

//  Vector properties

//  Function: int gsl_vector#typeext#_isnull (const gsl_vector#typeext# * v) 
//      This function returns 1 if all the elements of the vector v are zero, and 0 otherwise. };
	bool isnull(){return gsl_vector#typeext#_isnull (gsldata);}
};

// When you add create a view it will stick to its with the view until you call change_view
// ex:
// matrix_float m(5,5);
// vector_float v(5); 
// // ... 
// m.column(3) = v; //the 3rd column of the matrix m will equal v. 
class vector#typeext#_view : public vector#typeext#
{
 public:
	vector#typeext#_view(const vector#typeext#&     other) :vector#typeext#(){init(other);}
	vector#typeext#_view(const vector#typeext#_view& other):vector#typeext#(){init(other);}
	vector#typeext#_view(const gsl_vector#typeext#& gsl_other) : vector#typeext#() {init_with_gsl_vector(gsl_other);}

	void init(const vector#typeext#& other);
	void init_with_gsl_vector(const gsl_vector#typeext#& gsl_other);
	void change_view(const vector#typeext#& other){init(other);}
 private:
};

ostream& operator<< ( ostream& os, const vector#typeext# & vect );


// vector_type<>::type is a template interface to vector_?
// it is usefull for in templated situations for getting the correct vector type
#define tmp_type_is#typeext#
#ifdef tmp_type_is
typedef vector vector_double;
template<class T> 
struct vector_type  {typedef vector_double   type;};

template<class T> 
struct value_type  {typedef double   type;};

#else
template<> struct vector_type<#type#> {typedef vector#typeext# type;};
#endif
#undef tmp_type_is#typeext#

}
#endif// _vector_#type#_h
