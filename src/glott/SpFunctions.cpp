/*
 * SpFunctions.cpp
 *
 *  Created on: 3 Oct 2016
 *      Author: ljuvela
 */

/**
 * Function Interpolate_lin
 *
 * Interpolates linearly given vector to new vector of given length
 *
 * @param vector original vector
 * @param i_vector interpolated vector
 */

#include <gsl/gsl_spline.h>			/* GSL, Interpolation */
#include <gslwrap/vector_double.h>
#include <vector>
#include "definitions.h"

void Filter(const gsl::vector &b, const gsl::vector &a, const gsl::vector &x, gsl::vector *y) {
	int i,j;
	double sum;
	*y = gsl::vector(x.size(),true);

	/* Filter */
	for (i=0;i<(int)x.size();i++){
		sum = 0.0;
		for(j=0;j<(int)b.size();j++) { /* Loop for FIR filter */
			if ((i-j) >= 0) {
				sum += x(i-j)*b(j);
			}
		}
		for(j=1;j<(int)a.size();j++) { /* Loop for IIR filter */
			if ((i-j) >= 0) {
				sum -= (*y)(i-j)*a(j);
			}
		}
		(*y)(i) = sum;
	}
}

void Filter(const std::vector<double> &b, const std::vector<double> &a, const gsl::vector &x, gsl::vector *y) {
	size_t i;
	gsl::vector a_vec = gsl::vector(a.size());
	for(i=0;i<a_vec.size();i++)
		a_vec(i) = a[i];

	gsl::vector b_vec = gsl::vector(b.size());
	for(i=0;i<b_vec.size();i++)
		b_vec(i) = b[i];

	Filter(b_vec, a_vec, x, y);
}


void Filter(const gsl::vector b, const std::vector<double> &a, const gsl::vector &x, gsl::vector *y) {
	size_t i;
	gsl::vector a_vec = gsl::vector(a.size());
	for(i=0;i<a_vec.size();i++)
		a_vec(i) = a[i];

	Filter(b, a_vec, x, y);
}

void Filter(const std::vector<double> &b, const gsl::vector a, const gsl::vector &x, gsl::vector *y) {
	size_t i;
	gsl::vector b_vec = gsl::vector(b.size());
	for(i=0;i<b_vec.size();i++)
		b_vec(i) = b[i];

	Filter(b_vec, a, x, y);
}


void InterpolateLinear(const gsl::vector &vector, const size_t interpolated_size, gsl::vector *i_vector) {
	size_t len = vector.size();
	*i_vector = gsl::vector(interpolated_size);

	/* Read values to array */
	double x[len];
	double y[len];
	size_t i;
	for(i=0; i<len; i++) {
		x[i] = i;
		y[i] = vector(i);
	}
	gsl_interp_accel *acc = gsl_interp_accel_alloc();
	gsl_spline *spline = gsl_spline_alloc(gsl_interp_linear, len);
	gsl_spline_init(spline, x, y, len);
	double xi;
    i = 0;

    xi = x[0];
    while(i<interpolated_size) {
    	(*i_vector)(i) = gsl_spline_eval(spline, xi, acc);
    	xi += (len-1)/(double)(interpolated_size-1);
    	if(xi > len-1)
    		xi = len-1;
    	i++;
    }

    /* Free memory */
    gsl_spline_free(spline);
	gsl_interp_accel_free(acc);
}

void InterpolateNearest(const gsl::vector &vector, const size_t interpolated_size, gsl::vector *i_vector) {
	if(i_vector->is_set()) {
		if(i_vector->size() != interpolated_size) {
			i_vector->resize(interpolated_size);
		} else {
			i_vector->set_all(0.0);
		}
	} else {
		*i_vector = gsl::vector(interpolated_size);
	}
	int i,j;
	for (i=0;i<(int)i_vector->size();i++){
		j = rint((double)i/(double)(vector.size()-1)*(double)(i_vector->size()-1));
		j = GSL_MAX(0,j);
		j = GSL_MIN(j,(int)vector.size()-1);
		(*i_vector)(i) = vector(j);
	}
}

void ApplyWindowingFunction(const WindowingFunctionType &window_function, gsl::vector *frame) {
	size_t i;
	double n = (double)frame->size();
	switch(window_function) {
	case HANN :
		for(i=0;i<frame->size();i++)
			(*frame)(i) *= 0.5*(1.0-cos(2.0*M_PI*((double)i)/((n)-1.0)));
		break;
	case HAMMING :
		for(i=0;i<frame->size();i++)
			(*frame)(i) *= 0.53836 - 0.46164*(cos(2.0*M_PI*((double)i)/((n)-1.0)));
		break;
	case BLACKMAN :
		for(i=0;i<frame->size();i++)
			(*frame)(i) *= 0.42-0.5*cos(2.0*M_PI*((double)i)/((n)-1))+0.08*cos(4.0*M_PI*((double)i)/((n)-1.0));
		break;
	case COSINE :
		for(i=0;i<frame->size();i++)
			(*frame)(i) *= sqrt(0.5*(1.0-cos(2.0*M_PI*((double)i)/((n)-1.0))));
		break;
	}
}

void Autocorrelation(const gsl::vector &frame, const int &order, gsl::vector *r) {
	if(!r->is_set()) {
		*r = gsl::vector(order+1);
	} else {
		if((int)r->size() != order+1) {
			r->resize(order+1);
		} else {
			r->set_all(0.0);
		}
	}
	/* Autocorrelation sequence */
	int i,n;
	for(i=0; i<order+1;i++) {
		for(n=i; n<(int)frame.size(); n++) {
			(*r)(i) += frame(n)*frame(n-i);
		}
	}
}


void Levinson(const gsl::vector &r, gsl::vector *A) {
    size_t p = r.size()-1;
	if(!A->is_set()) {
		*A = gsl::vector(p+1);
	} else {
		if(A->size() != p+1) {
			A->resize(p+1);
		} else {
			A->set_all(0.0);
		}
	}

    gsl::vector tmp = gsl::vector(p+1,true);
    double e, ki;
    size_t i, j;

    /* Levinson-Durbin recursion for finding AR polynomial coefficients */
    e = r(0);
    (*A)(0) = 1.0;
    for(i = 1; i <= p; i++) {
        ki= 0.0;
        for(j = 1; j < i; j++) ki+= (*A)(j) * r(i-j);
        ki= (r(i) - ki) / e;
        (*A)(i) = ki;
        for(j = 1; j < i; j++) tmp(j) = (*A)(j) - ki * (*A)(i-j);
        for(j = 1; j < i; j++) (*A)(j) = tmp(j);
        e = (1 - ki* ki) * e;
    }
    (*A) *= -1.0; /* Invert coefficient signs */
    (*A)(0) = 1.0;
}


void AllPassDelay(const double &lambda, gsl::vector *signal) {
	double A[2] = {-lambda, 1.0};
	double B[2] = {0.0, lambda};

	int i,j;
	double sum;

	gsl::vector signal_orig(signal->size());
	signal_orig.copy(*signal);

	for(i=0;i<(int)signal->size();i++) {
		sum = 0.0;
		for(j=0;j<2;j++) {
			if((i-j) >= 0)
				sum += signal_orig(i-j)*A[j] + (*signal)(i-j)*B[j];
		}
		(*signal)(i) = sum;
	}
}

void FFT() {

}

/* Function AC_stabilize
 * Stabilize a polynomial by computing its autocorrelation and performing Levinson-Duribin recursion
 *  */
void AC_stabilize(gsl_vector *a,int Nframe) {
   int i;
   size_t N = a->size;
   size_t nfft = NextPow2(Nframe);
      double *data = calloc(nfft,sizeof(double)); /* complex, takes 2*nfft/2  values*/
      gsl_vector *X = gsl_vector_calloc(nfft/2+1);

      /* Calculate power spectral density */
      for (i=0; i<a->size; i++)
         data[i] = gsl_vector_get(a,i);
      gsl_fft_real_radix2_transform(data, 1, nfft);
      for(i=1; i<nfft/2; i++){
         gsl_vector_set(X, i, 1.0/(pow(data[i], 2) + pow(data[nfft-i], 2)));
      }
      gsl_vector_set(X, 0, 1.0/pow(data[0],2));
      gsl_vector_set(X,nfft/2,1.0/pow(data[nfft/2],2));

       /* Inverse transform PSD for autocorrelation */
       for (i=0;i<nfft/2+1;i++)
           data[i] = gsl_vector_get(X,i);
       for (i=nfft/2+1;i<nfft;i++)
           data[i] = 0;
       gsl_fft_halfcomplex_radix2_inverse(data, 1,nfft);

       /* Set values symmetrically to AC vector */
       gsl_vector *ac = gsl_vector_alloc(N);

       for(i=0;i<ac->size;i++)
         gsl_vector_set(ac,i,data[i]);
       /* Free memory*/

       Levinson(ac, a);
       free(data);
      gsl_vector_free(ac);
      gsl_vector_free(X);

}


