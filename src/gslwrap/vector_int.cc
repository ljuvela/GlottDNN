//  This matrix class is a C++ wrapper for the GNU Scientific Library

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

#include<gslwrap/vector_double.h>
#include<gslwrap/vector_int.h>

#define type_is_int
#ifdef type_is
#define type_is_double
#endif

namespace gsl
{
#ifndef __HP_aCC
	using namespace std;
//using std::string;
//using std::runtime_error;
#endif

//vector_int::create_vector_view( const gsl_vector_int_view &other )
vector_int_view 
vector_int::create_vector_view( const gsl_vector_int_view &other )
{
	vector_int view;
	view.gsldata = (gsl_vector_int*)malloc(sizeof(gsl_vector_int));
	*(view.gslobj()) = other.vector;
	view.gslobj()->owner = 0;
	return view;
}

void
vector_int::resize(size_t n)
{
	if (gsldata)
	{
		if (n==size())
			return;
		assert(gsldata->owner);
//  		if (!gsldata->owner)
//  		{
//  			cout << "vector_int::resize ERROR can't resize a vector view" << endl;
//  			exit(-1);
//  //			GSL_ERROR("You can't resize a vector view", GSL_EINVAL);
//  		}
		free();
	}
	alloc(n);
}

void 
vector_int::copy(const vector_int& other)
{
	if (this == &other)
		return;

	if (!other.is_set())
		gsldata = NULL;
	else
	{
		resize(other.size());
		gsl_vector_int_memcpy (gsldata,other.gsldata);
	}
}

bool 
vector_int::operator==(const vector_int& other) const 
{
	if (size() != other.size())
		return false;
	for (size_t i=0;i<size(); i++)
	{
		if (this->operator[](i) != other[i])
			return false;
	}
	return true;
}

vector_int_view 
vector_int::subvector (size_t offset, size_t n)
{
	gsl_vector_int_view view = gsl_vector_int_subvector (gsldata, offset, n);
	return vector_int_view::create_vector_view(view);
}

//vector_int_const_view 
const vector_int_view 
vector_int::subvector (size_t offset, size_t n) const 
{
	gsl_vector_int_view view = gsl_vector_int_subvector (gsldata, offset, n);
	return vector_int_view::create_vector_view(view);
}

// returns sum of all the elements.
int vector_int::sum() const 
{
   size_t i;
	int sum = 0;

	for ( i = 0; i < size(); i++ ) 
	{
		sum += gsl_vector_int_get(gsldata, i);
	}

	return( sum );
}

double 
vector_int::norm2() const
{
	vector t=*this;
	return gsl_blas_dnrm2(t.gslobj());
}

void vector_int::load( const char *filename )
{
   FILE * f = fopen( filename, "r" ) ;
//   vector_int temp;
   int size;
   ::fread(&size, sizeof(int), 1, f);
   gsldata=gsl_vector_int_alloc(size);   
   gsl_vector_int_fread ( f, gslobj() );
   fclose (f);
//   *this = temp;
}

void vector_int::save( const char *filename ) const
{
   FILE * f = fopen( filename, "w" ) ;
//   vector_int temp = *this;
   int s=size();
   ::fwrite(&s, sizeof(int), 1, f);
   gsl_vector_int_fwrite ( f, gslobj() );
   fclose ( f );
}


ostream& 
operator<< ( ostream& os, const vector_int & vect )
{
	os.setf( ios::fixed);
	for (size_t i=0;i<vect.size();i++)
	{
		os << vect[i] << endl;
	}
	return os;
}

//**************************************************************************
// Implementation of the vector_int_view class :
//**************************************************************************


void 
vector_int_view::init(const vector_int& other)
{
	free();
	gsldata = (gsl_vector_int*)malloc(sizeof(gsl_vector_int));
	*gsldata = *(other.gslobj());
	gsldata->owner = 0;
}

void 
vector_int_view::init_with_gsl_vector(const gsl_vector_int& gsl_other)
{
	free();
	gsldata = (gsl_vector_int*)malloc(sizeof(gsl_vector_int));
	*gsldata = gsl_other;
	gsldata->owner = 0;
}

}
