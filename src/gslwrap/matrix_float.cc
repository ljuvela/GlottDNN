
#include<gslwrap/matrix_double.h>
#include<gslwrap/matrix_float.h>
#include<gslwrap/vector_float.h>

#ifdef __HP_aCC //for aCC B3910B A.01.27
#include<iomanip.h>
#else //for gcc3
#include<iomanip>
#endif

#include <stdio.h>

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


matrix_float::matrix_float():m(NULL)
{
}

matrix_float::matrix_float( size_t rows, size_t cols , bool clear)
{
   if(clear){m = gsl_matrix_float_calloc( rows, cols );}
   else     {m = gsl_matrix_float_alloc ( rows, cols );}
}


matrix_float::~matrix_float()
{
   if ( m ) {gsl_matrix_float_free( m );}
}

//  matrix_float::matrix_float( const char *filename )
//  {
//     ;
//  }

void matrix_float::dimensions( size_t *num_rows, size_t *num_cols ) const
{
   *num_rows = m->size1;
   *num_cols = m->size2;
}


void matrix_float::set_elements( const float & new_value  )
{
   gsl_matrix_float_set_all( m, new_value );
}

void matrix_float::set_dimensions( size_t new_rows, size_t new_cols )
{
	if (!m)
	{
		m = gsl_matrix_float_calloc( new_rows, new_cols );
		return;
	}
	// if dimensions have changed re-allocate matrix
	else if ( (get_rows() != new_rows || get_cols() != new_cols )) {
		gsl_matrix_float_free( m );
		// allocate
		m = gsl_matrix_float_calloc( new_rows, new_cols );
	}
}


void matrix_float::load( const char *filename )
{
   FILE * f = fopen( filename, "r" ) ;
//   matrix_float temp;
   int rows;
   int cols;
   ::fread(&rows, sizeof(int), 1, f);
   ::fread(&cols, sizeof(int), 1, f);
   m = gsl_matrix_float_calloc( rows, cols );
   gsl_matrix_float_fread ( f, gslobj() );
   fclose (f);
//   *this = temp;
}

void matrix_float::save( const char *filename ) const
{
   FILE * f = fopen( filename, "w" ) ;
//   matrix_float temp = *this;
   int rows=get_rows();
   int cols=get_cols();
   ::fwrite(&rows, sizeof(int), 1, f);
   ::fwrite(&cols, sizeof(int), 1, f);
   gsl_matrix_float_fwrite ( f, gslobj() );
   fclose ( f );
}

ostream& operator<< ( ostream& os, const matrix_float & m )
{
   size_t i, j;

   os.setf( ios::fixed );

   for ( i = 0; i < m.get_rows(); i++ ) {
	   for ( j = 0; j < m.get_cols() - 1; j++ ) {
  		   os << m.get_element( i, j ) << " ";
	   }
	   os << m.get_element( i, j ) << endl;
   }

   return os;
}



bool matrix_float::operator==( const matrix_float &other ) const
{
   size_t i, j;
   
   // first check for same dimensions
   if ( size1() != other.size1() || size2() != other.size2() )
   {
      return false;
   }

   for ( i = 0; i < size1(); i++ ) {
      for ( j = 0; j < size2(); j++ ) {
         if ( this->get_element( i, j ) != other.get_element( i, j ) ) {
            return false;
         }
      }
   }

   return true;
}

matrix_float matrix_float::operator+( const matrix_float &other ) const
{
	matrix_float result(*this);
	gsl_matrix_float_add( result.m, other.m );
	return result;
}

matrix_float matrix_float::operator+( const float &f ) const
{
   matrix_float result( *this );
   gsl_matrix_float_add_constant( result.m, f );

   return( result );
}

matrix_float operator+( const float &f, const matrix_float &other )
{
   matrix_float result( other );
   gsl_matrix_float_add_constant( result.m, f );

   return( result );
}

matrix_float &matrix_float::operator+=( const float &f )
{
   gsl_matrix_float_add_constant( m, f );

   return( *this );
}

matrix_float &matrix_float::operator+=( const matrix_float &other )
{
   gsl_matrix_float_add( m, other.m );

   return( *this );
}

matrix_float matrix_float::operator-( const matrix_float &other ) const
{
   matrix_float result( *this );
   gsl_matrix_float_sub( result.m, other.m );

   return( result );
}

matrix_float matrix_float::operator-( const float &f ) const
{
   matrix_float result( *this );
   gsl_matrix_float_add_constant( result.m, -f );

   return( result );
}

matrix_float operator-( const float &f, const matrix_float &other )
{
   matrix_float result( -1 * other );
   gsl_matrix_float_add_constant( result.m, f );

   return( result );
}

matrix_float &matrix_float::operator-=( const float &f )
{
   gsl_matrix_float_add_constant( m, -f );

   return( *this );
}

matrix_float &matrix_float::operator-=( const matrix_float &other )
{
   gsl_matrix_float_sub( m, other.m );

   return( *this );
}


matrix_float matrix_float::operator*( const matrix_float &other ) const
{
	matrix result( get_rows(), other.get_cols() );
#ifdef type_is_double
	gsl_linalg_matmult(m,other.m,result.m);
#else //type_is_double
	matrix a=*this;
	matrix b=other;
	gsl_linalg_matmult(a.m,b.m,result.m);
#endif //type_is_double
	return result ;
}

matrix_float matrix_float::operator*( const float &f ) const
{
   matrix_float result( *this );
   gsl_matrix_float_scale( result.m, f );
   return( result );
}

matrix_float operator*( const float &f, const matrix_float &other )
{
   matrix_float result( other );
   gsl_matrix_float_scale( result.m, f );

   return( result );
}

matrix_float &matrix_float::operator*=( const float &f )
{
   gsl_matrix_float_scale( m, f );

   return( *this );
}

matrix_float &matrix_float::operator*=( const matrix_float &other )
{
   *this=(*this)*other;
   return( *this );
}

matrix_float matrix_float::operator/( const float &f ) const
{
   matrix_float result( *this );

   if ( f != 0.0 ) {
      gsl_matrix_float_scale( result.m, 1.0 / f );
   } else {
      cout << "e_r_r_o_r: division by zero." << endl;
      return( result );
   }

   return( result );
}

matrix_float &matrix_float::operator/=( const float &f )
{
   if ( f != 0.0 ) {
      gsl_matrix_float_scale( m, 1.0 / f );
   } else {
      cout << "e_rr_or: division by zero." << endl;
      return( *this );
   }

   return( *this );
}

matrix_float matrix_float::transpose() const 
{
   size_t i, j;
   matrix_float result( get_cols(), get_rows() );

   for ( i = 0; i < get_rows(); i++ ) {
      for ( j = 0; j < get_cols(); j++ ) {
         gsl_matrix_float_set( result.m, j, i, gsl_matrix_float_get( m, i, j ) );
      }
   }
   
   return( result );
}


matrix_float matrix_float::LU_decomp(gsl::permutation *perm,int *psign) const
{
	bool retPerm=perm!=NULL;
	if(!perm){perm = new permutation();}
	int sign;
	perm->resize( get_rows() );
	matrix result=*this;// this does conversion  if necessary
	gsl_linalg_LU_decomp(  result.m, perm->gsldata, &sign );

	if(!retPerm){delete perm;}
	if(psign){*psign=sign;}

   return result;// this does conversion  if necessary
}

matrix_float matrix_float::LU_invert() const
{
   permutation p;
   matrix a=*this;
   a=a.LU_decomp(&p);
   matrix inverse(size1(),size2());
   gsl_linalg_LU_invert( a.m, p.gsldata, inverse.m );
   return inverse;
}


// returns sum of all the elements.
float matrix_float::sum() const 
{
   size_t i, j;
	float sum = 0;

	for ( i = 0; i < get_rows(); i++ ) {
		for ( j = 0; j < get_cols(); j++ ) {
			sum += gsl_matrix_float_get( m, i, j );
		}
	}

	return( sum );
}

// returns logarithm of the determinant of the matrix.
double matrix_float::LU_lndet() const 
{
	matrix a=*this;
	a=a.LU_decomp();

	// calculate log determinant from LU decomposed matrix "a"
	return gsl_linalg_LU_lndet( a.m );
}

vector_float_view 
matrix_float::row( size_t rowindex )
{
	gsl_vector_float_view view=gsl_matrix_float_row(m, rowindex);
	return vector_float_view::create_vector_view(view);
}

const 
vector_float_view 
matrix_float::row( size_t rowindex ) const
{
	gsl_vector_float_view view=gsl_matrix_float_row(m, rowindex);
	return vector_float_view::create_vector_view(view);
}

vector_float_view 
matrix_float::column( size_t colindex )
{
	gsl_vector_float_view view=gsl_matrix_float_column(m, colindex);
	return vector_float_view::create_vector_view(view);
}

const 
vector_float_view 
matrix_float::column( size_t colindex ) const
{
	gsl_vector_float_view view=gsl_matrix_float_column(m, colindex);
	return vector_float_view::create_vector_view(view);
}

vector_float_view 
matrix_float::diagonal()
{
	gsl_vector_float_view view=gsl_matrix_float_diagonal(m);
	return vector_float_view::create_vector_view(view);
}

const 
vector_float_view 
matrix_float::diagonal() const
{
	gsl_vector_float_view view=gsl_matrix_float_diagonal(m);
	return vector_float_view::create_vector_view(view);
}

/** returns a row matrix_float containing a single row of the matrix. */
matrix_float matrix_float::get_row( size_t rowindex ) const 
{
	matrix_float rowmatrix( 1, get_cols() );
	gsl_vector_float *tempvector = gsl_vector_float_calloc( get_cols() );
	
	assert( rowindex > 0 && rowindex <= get_rows() );
//  	if ( rowindex < 0 || rowindex >= get_rows() )
//  	{
//  		cerr << "row index must be in range 0 to " << get_rows() - 1 << endl;
//  		exit( 1 );
//  	}

	gsl_matrix_float_get_row( tempvector, m, rowindex );
	gsl_matrix_float_set_row( rowmatrix.m, 0, tempvector );

	// tidy up
	gsl_vector_float_free( tempvector );
	
	return( rowmatrix );
}

/** returns a column matrix_float containing a single column of the matrix. */
matrix_float matrix_float::get_col( size_t colindex ) const 
{
	matrix_float columnmatrix( get_rows(), 1 );
	gsl_vector_float *tempvector = gsl_vector_float_calloc( get_rows() );
	
	assert( colindex > 0 && colindex <= get_cols() );
//  	if ( colindex < 0 || colindex >= get_cols() )
//  	{
//  		cerr << "column index must be in range 0 to " << get_cols() - 1 << endl;
//  		exit( 1 );
//  	}
	
	gsl_matrix_float_get_col( tempvector, m, colindex );
	gsl_matrix_float_set_col( columnmatrix.m, 0, tempvector );
	for ( size_t i = 0; i < get_rows(); i++ )
		cout << gsl_vector_float_get( tempvector, i ) << endl;

	// tidy up
	gsl_vector_float_free( tempvector );
	
	return( columnmatrix );
}

/** calculates sum of rows returned as a column matrix. */
matrix_float matrix_float::row_sum() const 
{
   size_t	i;
	matrix_float sum( get_rows(), 1 );
	
	sum.set_zero();
	for ( i = 0; i < get_rows(); i++ ) {
		sum.set_element( i, 0, get_row( i ).sum() );
	}
	
	return( sum );
}

/** calculates sum of columns returned as a row matrix. */
matrix_float matrix_float::column_sum() const 
{
	size_t	i;
	matrix_float sum( 1, get_cols() );
	
	sum.set_zero( );
	for ( i = 0; i < get_cols(); i++ ) {
		sum.set_element( 0, i, get_col( i ).sum() );
	}
		
	return( sum );
}

/** returns trace (diagonal sum) of a square matrix. */
double matrix_float::trace() const 
{
	size_t	i;
	double sum = 0.0;
	
	assert (get_rows() == get_cols() );
//  	if ( get_rows() != get_cols() ) {
//  		cerr << "e_r_r_o_r: cannot calculate trace of non-square matrix.";
//  		cerr << endl;
//  		exit( 1 );
//  	}

	// calculate sum of diagonal elements
	for ( i = 0; i < get_rows(); i++ ) {
		sum += get_element( i, i );
	}

	return( sum );
}

/** calculates cholesky decomposition of the matrix, returning success if matrix_float is positive definite. */
// don't forget to de-allocate a, which is allocated in this method.
int matrix_float::cholesky_decomp( matrix_float &a ) const 
{
	//int i, j;
	int error;
	matrix result=*this;
	// do decomposition with call to g_s_l
	error = gsl_linalg_cholesky_decomp( result.m );
	a=result;
	return error;
}

/** diag operator (sets the diagonal elements of the matrix to the elements of v */
void matrix_float::diag(const vector_float& v)
{
	int dim=v.size();
	set_dimensions(dim, dim);
	set_zero();
	for (int i=0;i<dim;i++)
		set_element(i,i,v(i));
}


/** sets matrix_float to a k dimensional unit matrix. */
void matrix_float::identity( size_t k ) 
{
	set_dimensions( k, k );
	set_zero();
	set_diagonal( 1 );
}

/** set diagonal elements of a square matrix_float to f. */
void matrix_float::set_diagonal( float f ) 
{
	size_t i;
	size_t mn=(get_rows()<get_cols() ? get_rows() : get_cols());
	for ( i = 0; i < mn; i++ )
			set_element( i, i, f );
}

/** returns tru if matrix_float is square, false otherwise. */
bool matrix_float::is_square() const 
{
	if ( get_rows() == get_cols() )
		return true;
	return false;
}

/** returns sum of nth power of all elements. */
double matrix_float::norm( double n ) const 
{
	size_t i, j;
	double sum = 0.0;
	
	for ( i = 0; i < get_rows(); i++ ) {
		for ( j = 0; j < get_cols(); j++ ) {
#ifdef type_is_float // for HP aCC
			sum += pow( (double)(get_element( i, j )), n );
#else
			sum += pow( get_element( i, j ), n );
#endif
		}
	}
	
	return sum;
}

}
