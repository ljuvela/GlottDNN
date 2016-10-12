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
#include<gslwrap/vector_float.h>

#define type_is_float
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

//vector_float::create_vector_view( const gsl_vector_float_view &other )
vector_float_view 
vector_float::create_vector_view( const gsl_vector_float_view &other )
{
	vector_float view;
	view.gsldata = (gsl_vector_float*)malloc(sizeof(gsl_vector_float));
	*(view.gslobj()) = other.vector;
	view.gslobj()->owner = 0;
	return view;
}

void
vector_float::resize(size_t n)
{
	if (gsldata)
	{
		if (n==size())
			return;
		assert(gsldata->owner);
//  		if (!gsldata->owner)
//  		{
//  			cout << "vector_float::resize ERROR can't resize a vector view" << endl;
//  			exit(-1);
//  //			GSL_ERROR("You can't resize a vector view", GSL_EINVAL);
//  		}
		free();
	}
	alloc(n);
}

void 
vector_float::copy(const vector_float& other)
{
	if (this == &other)
		return;

	if (!other.is_set())
		gsldata = NULL;
	else
	{
		resize(other.size());
		gsl_vector_float_memcpy (gsldata,other.gsldata);
	}
}

bool 
vector_float::operator==(const vector_float& other) const 
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

vector_float_view 
vector_float::subvector (size_t offset, size_t n)
{
	gsl_vector_float_view view = gsl_vector_float_subvector (gsldata, offset, n);
	return vector_float_view::create_vector_view(view);
}

//vector_float_const_view 
const vector_float_view 
vector_float::subvector (size_t offset, size_t n) const 
{
	gsl_vector_float_view view = gsl_vector_float_subvector (gsldata, offset, n);
	return vector_float_view::create_vector_view(view);
}

// returns sum of all the elements.
float vector_float::sum() const 
{
   size_t i;
	float sum = 0;

	for ( i = 0; i < size(); i++ ) 
	{
		sum += gsl_vector_float_get(gsldata, i);
	}

	return( sum );
}

double 
vector_float::norm2() const
{
	vector t=*this;
	return gsl_blas_dnrm2(t.gslobj());
}

void vector_float::load( const char *filename )
{
   FILE * f = fopen( filename, "r" ) ;
//   vector_float temp;
   int size;
   ::fread(&size, sizeof(int), 1, f);
   gsldata=gsl_vector_float_alloc(size);   
   gsl_vector_float_fread ( f, gslobj() );
   fclose (f);
//   *this = temp;
}

void vector_float::save( const char *filename ) const
{
   FILE * f = fopen( filename, "w" ) ;
//   vector_float temp = *this;
   int s=size();
   ::fwrite(&s, sizeof(int), 1, f);
   gsl_vector_float_fwrite ( f, gslobj() );
   fclose ( f );
}


ostream& 
operator<< ( ostream& os, const vector_float & vect )
{
	os.setf( ios::fixed);
	for (size_t i=0;i<vect.size();i++)
	{
		os << vect[i] << endl;
	}
	return os;
}

//**************************************************************************
// Implementation of the vector_float_view class :
//**************************************************************************


void 
vector_float_view::init(const vector_float& other)
{
	free();
	gsldata = (gsl_vector_float*)malloc(sizeof(gsl_vector_float));
	*gsldata = *(other.gslobj());
	gsldata->owner = 0;
}

void 
vector_float_view::init_with_gsl_vector(const gsl_vector_float& gsl_other)
{
	free();
	gsldata = (gsl_vector_float*)malloc(sizeof(gsl_vector_float));
	*gsldata = gsl_other;
	gsldata->owner = 0;
}

}
