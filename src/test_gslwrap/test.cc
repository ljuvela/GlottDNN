#ifdef __HP_aCC //for aCC B3910B A.01.27
#include <iostream.h>
#else //for gcc3
#include <iostream>
#endif

#include<gslwrap/matrix_float.h>
#include<gslwrap/matrix_double.h>
#include<gslwrap/vector_float.h>
#include<gslwrap/vector_int.h>
#include<gslwrap/vector_double.h>
#include<gslwrap/random_generator.h>
#include<gslwrap/histogram.h>
using namespace gsl;

#ifndef __HP_aCC
	using namespace std;
//using std::string;
//using std::runtime_error;
#endif

void 
Histogram()
{
	// Generate a big number of uniform samples
	random_generator generator; // default constructor takes seed from ENV variable
  	int nSamples=10000;
  	gsl::vector samples(nSamples);
	int i;
  	for (i=0;i<nSamples;i++)
  		samples[i] = generator.uniform();

	// histogram:
	histogram h(50, 0, 1);
	for (i=0;i<nSamples;i++)
		h.increment(samples[i]);

	double maxval=h.max_val();
	double minval=h.min_val();
	double diff=(maxval - minval)/(double)nSamples;
	cout << "Difference (maxval - minval)/nSamples: " <<  diff << endl;
	if (diff > 0.01)
		cout << "test_histogram did NOT pass!" << endl;
	else 
		cout << "test_histogram passed!" << endl;
//  	double m=h.mean();
//  	cout << "Mean=" << m << endl;
	
}

void
Vector()
{
	gsl::vector vd(10);
	gsl::vector_float  vf(10);
	gsl::vector_int    vi(10);
	vd.set_all(0.0);
	if (!vd.isnull())
		cout << "test_vector ERROR" << endl;
	
	vd.set_all(1.0);
	double norm=vd.norm2();
	cout << "norm=" << norm << endl;
}

// Testing  creation, dimensioning and assigning functions:
void 
VectorFloat()
{
	vector_float test;
	vector_float mal(10);
	for (int i=0; i<10; i++)
	{mal[i] = i;}

	test = mal;
  	cout << "Test vector should be:" << endl <<mal<< endl;
  	cout << "Test vector is:" << endl <<test<< endl;
	
	if (mal != test)
		cout << "vector_float ERROR Test did not pass !! --------------" << endl;
	else 
		cout << "vector_float Test passed !! --------------" << endl;
}

void
VectorView()
{
	cout << "VectorViewTest --------------------------" << endl;
	int i,j;
	int size=5;
	matrix_float m(size, size);
	matrix_float ver(size, size);ver.set_all(5.0);
	vector_float v(size);
	for (i=0;i<size;i++)
	{
		v[i]=size-(i+1);
		for (j=0;j<size;j++)
			m(i,j)=(i+1);
	}
	cout << "m=" << endl << m << endl;
	cout << "v=" << endl << v << endl;

	vector_float_view col_viewer = m.column(3);
	vector_float_view viewer = m.row(3);
	cout << "viewer (3.row before adding)=" << endl << viewer << endl;
	cout << "col_viewer (3.col before adding)=" << endl << col_viewer << endl;
	for (i=0;i<m.get_cols();i++)
		m.column(i) += v;

	cout << "viewer (3.row after adding) =" << endl << viewer << endl;
	cout << "col_viewer (3.col after adding)=" << endl << col_viewer << endl;
	cout << "\"m+v\"" << endl << m << endl;
	
	col_viewer.change_view(viewer);
	cout << "col_viewer after changing to viewer=" << endl << col_viewer << endl;

	if (m!=ver || m.column(0)!=viewer || col_viewer!=viewer) 
		cout << "VectorViewTest failed !! ----------------" << endl;
	else 
		cout << "VectorViewTest passed !! ----------------" << endl;
	
}

void VectorDiagonalView()
{
	matrix m(5, 10);
//  	vector_view v = m.diagonal();
//  	v.set_all(1.0);
	m.diagonal()+=1.0;//.set_all(1.0);
//	cout << "m=" << m << endl;
	bool pass=true;
	for (int i=0;i<5;i++)
	{
		if(m(i,i)!=1.0)
			pass=false;
	}
	if (!pass) 
		cout << "VectorDiagonalViewTest failed !! ----------------" << endl;
	else 
		cout << "VectorDiagonalViewTest passed !! ----------------" << endl;
}

void VectorView2()
{
	cout << "VectorViewTest2 ----------------------------" << endl;
	matrix m1(10,10);
	matrix m2(10,10);
	gsl::vector v(10);
	int i;

	for (i=0;i<10; i++){v[i]=i*2;}

	for (i=0;i<10; i++)
	{
		m1.column(i) = v;
		m2.column(i) = v;
	}

	for (i=0;i<10; i++)
		m1.column(i) -= m2.column(i);

	cout << "Sum m1-m2 (m1==m2) : " << m1.sum() << endl;
	if (m1.sum())
	{
		cout << "VectorViewTest2 Failed !! (operator -= ?) " << endl;
		exit(-1);
	}
	else 
		cout << "VectorViewTest2 Passed !! -------------------------" << endl;
}

void VectorView3()
{
	gsl::vector v(10);
	gsl::vector_view view1 = v;
	gsl::vector_view view2 = v;
	view2 = view1;
}
//some test calls to gsl:
void 
GSLFunctionCall()
{
	int size=5;
	matrix m(size,size);
	for (int i=0;i<size;i++)
		for (int j=0;j<size;j++)
			m(i,j) = (i+1)*(j+1);
	double lndet = gsl_linalg_LU_lndet(m.gslobj());
	cout << "m = " << m << endl;
	cout << "lndet(m) = " << lndet << endl;

	gsl::vector sol(size);
	for (int k=0;k<size;k++)
		sol[k]=k+1;
	cout << "Solution of " << m << " x= " << sol <<endl;

	gsl::vector tau(size);
	gsl_linalg_QR_decomp(m.gslobj(), tau.gslobj());
	gsl_linalg_QR_svx(m.gslobj(), tau.gslobj(), sol.gslobj());
	
	cout << "sol= " << sol << endl;
}

void
RandomNumberGenerator()
{
	cout << "RandomNumberGeneratorTest -------------------" << endl;
	random_generator generator; // default constructor takes seed from ENV variable
  	int nSamples=10000;
  	gsl::vector samples(nSamples);
  	for (int i=0;i<nSamples;i++)
  		samples[i] = generator.uniform();

	double mean=samples.sum()/(float)nSamples;
  	cout << "Mean of sampling : " <<  mean << endl;
	// should be close to 0.5
	if (mean <0.51 && mean >0.49)
		cout << "RandomNumberGeneratorTest Passed !! -------------" << endl;
	else 
	{
		cout << "RandomNumberGeneratorTest Failed !! -------------" << endl;
		exit(-1);
	}
}

void LUInvertAndDecomp()
{
     matrix a( 3, 3 );
     matrix b( 3, 3 );
     srand48(10);

     int i, j;
     for(i=0;i<a.size1();i++)
  	   for(j=0;j<a.size2();j++) a(i,j)=drand48();
     for(i=0;i<b.size1();i++)
  	   for(j=0;j<b.size2();j++) b(i,j)=drand48();

     cout << "a:" << endl << a << endl;
     cout << "b:" << endl << b << endl;

     cout << "a + b:" << endl << a + b << endl;
     cout << "a - b:" << endl << a - b << endl;
     cout << "a + 1.0:" << endl << a + 1.0 << endl;
     cout << "1.0 - a:" << endl << 1.0 - a << endl;
     cout << "a * 10.0:" << endl << a * 10.0 << endl;
     cout << "10.0 * a:" << endl << 10.0 * a << endl;
     cout << "a / 10.0:" << endl << a / 10.0 << endl;
     cout << "a * b:" << endl << a * b << endl;
     cout << "a.LU_decomp():" << endl << a.LU_decomp() << endl;
     cout << "a.LU_invert():" << endl << a.LU_invert() << endl;
     cout << "a.LU_invert() * a:" << endl << a.LU_invert() * a << endl;
} 

