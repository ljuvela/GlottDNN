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
#include<gslwrap/vector_#type#.h>

#define type_is#typeext#
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

//vector#typeext#::create_vector_view( const gsl_vector#typeext#_view &other )
vector#typeext#_view 
vector#typeext#::create_vector_view( const gsl_vector#typeext#_view &other )
{
	vector#typeext# view;
	view.gsldata = (gsl_vector#typeext#*)malloc(sizeof(gsl_vector#typeext#));
	*(view.gslobj()) = other.vector;
	view.gslobj()->owner = 0;
	return view;
}

void
vector#typeext#::resize(size_t n)
{
	if (gsldata)
	{
		if (n==size())
			return;
		assert(gsldata->owner);
//  		if (!gsldata->owner)
//  		{
//  			cout << "vector#typeext#::resize ERROR can't resize a vector view" << endl;
//  			exit(-1);
//  //			GSL_ERROR("You can't resize a vector view", GSL_EINVAL);
//  		}
		free();
	}
	alloc(n);
}

void 
vector#typeext#::copy(const vector#typeext#& other)
{
	if (this == &other)
		return;

	if (!other.is_set())
		gsldata = NULL;
	else
	{
		resize(other.size());
		gsl_vector#typeext#_memcpy (gsldata,other.gsldata);
	}
}

bool 
vector#typeext#::operator==(const vector#typeext#& other) const 
{
	if (size() != other.size())
		return false;
	for (int i=0;i<size(); i++)
	{
		if (this->operator[](i) != other[i])
			return false;
	}
	return true;
}

vector#typeext#_view 
vector#typeext#::subvector (size_t offset, size_t n)
{
	gsl_vector#typeext#_view view = gsl_vector#typeext#_subvector (gsldata, offset, n);
	return vector#typeext#_view::create_vector_view(view);
}

//vector#typeext#_const_view 
const vector#typeext#_view 
vector#typeext#::subvector (size_t offset, size_t n) const 
{
	gsl_vector#typeext#_view view = gsl_vector#typeext#_subvector (gsldata, offset, n);
	return vector#typeext#_view::create_vector_view(view);
}

// returns sum of all the elements.
#type# vector#typeext#::sum() const 
{
	int i;
	#type# sum = 0;

	for ( i = 0; i < size(); i++ ) 
	{
		sum += gsl_vector#typeext#_get(gsldata, i);
	}

	return( sum );
}

double 
vector#typeext#::norm2() const
{
	vector t=*this;
	return gsl_blas_dnrm2(t.gslobj());
}

void vector#typeext#::load( const char *filename )
{
   FILE * f = fopen( filename, "r" ) ;
//   vector#typeext# temp;
   int size;
   ::fread(&size, sizeof(int), 1, f);
   gsldata=gsl_vector#typeext#_alloc(size);   
   gsl_vector#typeext#_fread ( f, gslobj() );
   fclose (f);
//   *this = temp;
}

void vector#typeext#::save( const char *filename ) const
{
   FILE * f = fopen( filename, "w" ) ;
//   vector#typeext# temp = *this;
   int s=size();
   ::fwrite(&s, sizeof(int), 1, f);
   gsl_vector#typeext#_fwrite ( f, gslobj() );
   fclose ( f );
}


ostream& 
operator<< ( ostream& os, const vector#typeext# & vect )
{
	os.setf( ios::fixed);
	for (int i=0;i<vect.size();i++)
	{
		os << vect[i] << endl;
	}
	return os;
}

//**************************************************************************
// Implementation of the vector#typeext#_view class :
//**************************************************************************


void 
vector#typeext#_view::init(const vector#typeext#& other)
{
	free();
	gsldata = (gsl_vector#typeext#*)malloc(sizeof(gsl_vector#typeext#));
	*gsldata = *(other.gslobj());
	gsldata->owner = 0;
}

void 
vector#typeext#_view::init_with_gsl_vector(const gsl_vector#typeext#& gsl_other)
{
	free();
	gsldata = (gsl_vector#typeext#*)malloc(sizeof(gsl_vector#typeext#));
	*gsldata = gsl_other;
	gsldata->owner = 0;
}

}
