
#include<gslwrap/matrix_double.h>
#include<gslwrap/matrix_double.h>
#include<gslwrap/vector_double.h>

#ifdef __HP_aCC //for aCC B3910B A.01.27
#include<iomanip.h>
#else //for gcc3
#include<iomanip>
#endif

#include <stdio.h>

#define type_is
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


matrix::matrix():m(NULL)
{
}

matrix::matrix( size_t rows, size_t cols , bool clear)
{
   if(clear){m = gsl_matrix_calloc( rows, cols );}
   else     {m = gsl_matrix_alloc ( rows, cols );}
}


matrix::~matrix()
{
   if ( m ) {gsl_matrix_free( m );}
}

//  matrix::matrix( const char *filename )
//  {
//     ;
//  }

void matrix::dimensions( size_t *num_rows, size_t *num_cols ) const
{
   *num_rows = m->size1;
   *num_cols = m->size2;
}


void matrix::set_elements( const double & new_value  )
{
   gsl_matrix_set_all( m, new_value );
}

void matrix::set_dimensions( size_t new_rows, size_t new_cols )
{
	if (!m)
	{
		m = gsl_matrix_calloc( new_rows, new_cols );
		return;
	}
	// if dimensions have changed re-allocate matrix
	else if ( (get_rows() != new_rows || get_cols() != new_cols )) {
		gsl_matrix_free( m );
		// allocate
		m = gsl_matrix_calloc( new_rows, new_cols );
	}
}


void matrix::load( const char *filename )
{
   FILE * f = fopen( filename, "r" ) ;
//   matrix temp;
   int rows;
   int cols;
   ::fread(&rows, sizeof(int), 1, f);
   ::fread(&cols, sizeof(int), 1, f);
   m = gsl_matrix_calloc( rows, cols );
   gsl_matrix_fread ( f, gslobj() );
   fclose (f);
//   *this = temp;
}

void matrix::save( const char *filename ) const
{
   FILE * f = fopen( filename, "w" ) ;
//   matrix temp = *this;
   int rows=get_rows();
   int cols=get_cols();
   ::fwrite(&rows, sizeof(int), 1, f);
   ::fwrite(&cols, sizeof(int), 1, f);
   gsl_matrix_fwrite ( f, gslobj() );
   fclose ( f );
}

ostream& operator<< ( ostream& os, const matrix & m )
{
   size_t i, j;

   os.setf( ios::fixed );

//FIXME for aCC (doesn't find correct outstream function
//     for ( i = 0; i < m.get_rows(); i++ ) {
//    	   for ( j = 0; j < m.get_cols() - 1; j++ ) {
//  		   os << setprecision( 6 ) << setw( 11 ) ;//<< m.get_element( i, j ) << " ";
//    	   }
//    	   os << setprecision( 6 ) << setw( 11 ) ;//<< m.get_element( i, j ) << endl;
//     }

   for ( i = 0; i < m.get_rows(); i++ ) {
	   for ( j = 0; j < m.get_cols() - 1; j++ ) {
  		   os << m.get_element( i, j ) << " ";
	   }
	   os << m.get_element( i, j ) << endl;
   }

   return os;
}



void matrix::load_binary( const char *filename )
{
   ;
}

void matrix::save_binary( const char *filename ) const
{
   ;
}

bool matrix::operator==( const matrix &other ) const
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

matrix matrix::operator+( const matrix &other ) const
{
	matrix result(*this);
	gsl_matrix_add( result.m, other.m );
	return result;
}

matrix matrix::operator+( const double &f ) const
{
   matrix result( *this );
   gsl_matrix_add_constant( result.m, f );

   return( result );
}

matrix operator+( const double &f, const matrix &other )
{
   matrix result( other );
   gsl_matrix_add_constant( result.m, f );

   return( result );
}

matrix &matrix::operator+=( const double &f )
{
   gsl_matrix_add_constant( m, f );

   return( *this );
}

matrix &matrix::operator+=( const matrix &other )
{
   gsl_matrix_add( m, other.m );

   return( *this );
}

matrix matrix::operator-( const matrix &other ) const
{
   matrix result( *this );
   gsl_matrix_sub( result.m, other.m );

   return( result );
}

matrix matrix::operator-( const double &f ) const
{
   matrix result( *this );
   gsl_matrix_add_constant( result.m, -f );

   return( result );
}

matrix operator-( const double &f, const matrix &other )
{
   matrix result( -1 * other );
   gsl_matrix_add_constant( result.m, f );

   return( result );
}

matrix &matrix::operator-=( const double &f )
{
   gsl_matrix_add_constant( m, -f );

   return( *this );
}

matrix &matrix::operator-=( const matrix &other )
{
   gsl_matrix_sub( m, other.m );

   return( *this );
}


matrix matrix::operator*( const matrix &other ) const
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

matrix matrix::operator*( const double &f ) const
{
   matrix result( *this );
   gsl_matrix_scale( result.m, f );
   return( result );
}

matrix operator*( const double &f, const matrix &other )
{
   matrix result( other );
   gsl_matrix_scale( result.m, f );

   return( result );
}

matrix &matrix::operator*=( const double &f )
{
   gsl_matrix_scale( m, f );

   return( *this );
}

matrix &matrix::operator*=( const matrix &other )
{
   *this=(*this)*other;
   return( *this );
}

matrix matrix::operator/( const double &f ) const
{
   matrix result( *this );

   if ( f != 0.0 ) {
      gsl_matrix_scale( result.m, 1.0 / f );
   } else {
      cout << "e_r_r_o_r: division by zero." << endl;
      return( result );
   }

   return( result );
}

matrix &matrix::operator/=( const double &f )
{
   if ( f != 0.0 ) {
      gsl_matrix_scale( m, 1.0 / f );
   } else {
      cout << "e_rr_or: division by zero." << endl;
      return( *this );
   }

   return( *this );
}

matrix matrix::transpose() const 
{
   int i, j;
   matrix result( get_cols(), get_rows() );

   for ( i = 0; i < get_rows(); i++ ) {
      for ( j = 0; j < get_cols(); j++ ) {
         gsl_matrix_set( result.m, j, i, gsl_matrix_get( m, i, j ) );
      }
   }
   
   return( result );
}


matrix matrix::LU_decomp(gsl::permutation *perm,int *psign) const
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

matrix matrix::LU_invert() const
{
   int i, j;
   permutation p;
   matrix a=*this;
   a=a.LU_decomp(&p);
   matrix inverse(size1(),size2());
   gsl_linalg_LU_invert( a.m, p.gsldata, inverse.m );
   return inverse;
}


// returns sum of all the elements.
double matrix::sum() const 
{
	int i, j;
	double sum = 0;

	for ( i = 0; i < get_rows(); i++ ) {
		for ( j = 0; j < get_cols(); j++ ) {
			sum += gsl_matrix_get( m, i, j );
		}
	}

	return( sum );
}

// returns logarithm of the determinant of the matrix.
double matrix::LU_lndet() const 
{
	matrix a=*this;
	a=a.LU_decomp();

	// calculate log determinant from LU decomposed matrix "a"
	return gsl_linalg_LU_lndet( a.m );
}

vector_view 
matrix::row( size_t rowindex )
{
	gsl_vector_view view=gsl_matrix_row(m, rowindex);
	return vector_view::create_vector_view(view);
}

const 
vector_view 
matrix::row( size_t rowindex ) const
{
	gsl_vector_view view=gsl_matrix_row(m, rowindex);
	return vector_view::create_vector_view(view);
}

vector_view 
matrix::column( size_t colindex )
{
	gsl_vector_view view=gsl_matrix_column(m, colindex);
	return vector_view::create_vector_view(view);
}

const 
vector_view 
matrix::column( size_t colindex ) const
{
	gsl_vector_view view=gsl_matrix_column(m, colindex);
	return vector_view::create_vector_view(view);
}

vector_view 
matrix::diagonal()
{
	gsl_vector_view view=gsl_matrix_diagonal(m);
	return vector_view::create_vector_view(view);
}

const 
vector_view 
matrix::diagonal() const
{
	gsl_vector_view view=gsl_matrix_diagonal(m);
	return vector_view::create_vector_view(view);
}


void matrix::set_row_vec(const size_t rowindex, const vector &vec) {
	assert( rowindex >= 0 && rowindex < get_rows() );
	assert(vec.size() == get_cols());

	size_t i;
	for(i=0; i<get_cols(); i++)
		gsl_matrix_set(m,rowindex,i,vec(i));
}

void matrix::set_col_vec(const size_t colindex, const vector &vec) {
	assert( colindex >= 0 && colindex < get_cols() );
	assert(vec.size() == get_rows());

	size_t i;
	for(i=0; i<get_rows(); i++)
		gsl_matrix_set(m,i,colindex,vec(i));
}


/** returns a vector containing a single row of the matrix. */
vector matrix::get_row_vec( size_t rowindex ) const {
	vector rowvector(get_cols());
	assert( rowindex >= 0 && rowindex < get_rows() );

	size_t i;
	for(i=0; i<get_cols(); i++)
		rowvector(i) = gsl_matrix_get(m,rowindex,i);

	return rowvector;
}

/** returns a vector containing a single column of the matrix. */
vector matrix::get_col_vec( size_t colindex ) const {
	vector colvector(get_rows());
	assert( colindex >= 0 && colindex < get_cols() );

	size_t i;
	for(i=0; i<get_rows(); i++)
		colvector(i) = gsl_matrix_get(m,i,colindex);

	return colvector;
}


/** returns a row matrix containing a single row of the matrix. */
matrix matrix::get_row( size_t rowindex ) const 
{
	matrix rowmatrix( 1, get_cols() );
	gsl_vector *tempvector = gsl_vector_calloc( get_cols() );
	
	assert( rowindex > 0 && rowindex <= get_rows() );
//  	if ( rowindex < 0 || rowindex >= get_rows() )
//  	{
//  		cerr << "row index must be in range 0 to " << get_rows() - 1 << endl;
//  		exit( 1 );
//  	}

	gsl_matrix_get_row( tempvector, m, rowindex );
	gsl_matrix_set_row( rowmatrix.m, 0, tempvector );

	// tidy up
	gsl_vector_free( tempvector );
	
	return( rowmatrix );
}

/** returns a column matrix containing a single column of the matrix. */
matrix matrix::get_col( size_t colindex ) const 
{
	matrix columnmatrix( get_rows(), 1 );
	gsl_vector *tempvector = gsl_vector_calloc( get_rows() );
	
	assert( colindex > 0 && colindex <= get_cols() );
//  	if ( colindex < 0 || colindex >= get_cols() )
//  	{
//  		cerr << "column index must be in range 0 to " << get_cols() - 1 << endl;
//  		exit( 1 );
//  	}
	
	gsl_matrix_get_col( tempvector, m, colindex );
	gsl_matrix_set_col( columnmatrix.m, 0, tempvector );
	for ( int i = 0; i < get_rows(); i++ )
		cout << gsl_vector_get( tempvector, i ) << endl;

	// tidy up
	gsl_vector_free( tempvector );
	
	return( columnmatrix );
}

/** calculates sum of rows returned as a column matrix. */
matrix matrix::row_sum() const 
{
	int	i;
	matrix sum( get_rows(), 1 );
	
	sum.set_zero();
	for ( i = 0; i < get_rows(); i++ ) {
		sum.set_element( i, 0, get_row( i ).sum() );
	}
	
	return( sum );
}

/** calculates sum of columns returned as a row matrix. */
matrix matrix::column_sum() const 
{
	int	i;
	matrix sum( 1, get_cols() );
	
	sum.set_zero( );
	for ( i = 0; i < get_cols(); i++ ) {
		sum.set_element( 0, i, get_col( i ).sum() );
	}
		
	return( sum );
}

/** returns trace (diagonal sum) of a square matrix. */
double matrix::trace() const 
{
	int	i;
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

/** calculates cholesky decomposition of the matrix, returning success if matrix is positive definite. */
// don't forget to de-allocate a, which is allocated in this method.
int matrix::cholesky_decomp( matrix &a ) const 
{
	int i, j;
	int error;
	matrix result=*this;
	// do decomposition with call to g_s_l
	error = gsl_linalg_cholesky_decomp( result.m );
	a=result;
	return error;
}

/** diag operator (sets the diagonal elements of the matrix to the elements of v */
void matrix::diag(const vector& v)
{
	int dim=v.size();
	set_dimensions(dim, dim);
	set_zero();
	for (int i=0;i<dim;i++)
		set_element(i,i,v(i));
}


/** sets matrix to a k dimensional unit matrix. */
void matrix::identity( size_t k ) 
{
	set_dimensions( k, k );
	set_zero();
	set_diagonal( 1 );
}

/** set diagonal elements of a square matrix to f. */
void matrix::set_diagonal( double f ) 
{
	size_t i;
	int mn=(get_rows()<get_cols() ? get_rows() : get_cols());
	for ( i = 0; i < mn; i++ )
			set_element( i, i, f );
}

/** returns tru if matrix is square, false otherwise. */
bool matrix::is_square() const 
{
	if ( get_rows() == get_cols() )
		return true;
	return false;
}

/** returns sum of nth power of all elements. */
double matrix::norm( double n ) const 
{
	int i, j;
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
