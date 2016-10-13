
#include <gsl/gsl_spline.h>			/* GSL, Interpolation */
#include <gsl/gsl_fft_real.h>
#include <gsl/gsl_fft_halfcomplex.h>
#include <gsl/gsl_errno.h>       /* GSL, Error handling */
#include <gsl/gsl_poly.h>        /* GSL, Polynomials */
#include <gsl/gsl_sort_double.h> /* GSL, Sort double */
#include <gsl/gsl_sort_vector.h> /* GSL, Sort vector */
#include <gsl/gsl_complex.h>     /* GSL, Complex numbers */
#include <gsl/gsl_complex_math.h>   /* GSL, Arithmetic operations for complex numbers */
#include <gslwrap/vector_double.h>
#include <vector>
#include "ComplexVector.h"
#include "definitions.h"
#include "SpFunctions.h"
#include "QmfFunctions.h"



gsl::vector Qmf::LoadFilter(const std::vector<double> &saved_filter) {
   gsl::vector filter(saved_filter.size());
   size_t i;
   for(i=0;i<filter.size();i++)
      filter(i) = saved_filter[i];

   return filter;
}

gsl::vector Qmf::GetMatchingFilter(const gsl::vector &H0) {
	size_t i;
   gsl::vector H1(H0.size());
	for(i=0;i<H0.size();i++) {
		H1(i) = H0(i)*pow(-1,(1+i)%2);
	}
	return H1;
}

void Qmf::GetSubBands(const gsl::vector &frame, const gsl::vector &H0, const gsl::vector &H1,
                     gsl::vector *frame_qmf1, gsl::vector *frame_qmf2) {
	int L = (int)H0.size();
	int N = (int)frame.size();
	int i;
	gsl::vector tempframe(2*L+N,true);
	gsl::vector x21(2*L+N,true);
	gsl::vector x22(2*L+N,true);
   gsl::vector B(1);B(0)=1.0;

	// Set delay line for filtering
	for(i=0;i<N;i++)
		tempframe(i+L) = frame(i);

	// Lowpass & Highpass filtering with QMF filters
	Filter(H0,B,tempframe,&x21);
	Filter(H1,B,tempframe,&x22);

	// Downsampling
	int start = L/2;
	int stop = start+N;
	int ii = 0;
	for(i=start;i<stop;i=i+2) {
		(*frame_qmf1)(ii) = x21(i);
		(*frame_qmf2)(ii) = x22(i);
		ii++;
	}
}

void Qmf::Decimate(const gsl::vector &frame_orig, const int skip, gsl::vector *frame_decimated) {
   frame_decimated->set_all(0.0);
   size_t i;
   for(i=0;i<frame_decimated->size();i++) {
      if(skip*i >= frame_orig.size())
         break;
      (*frame_decimated)(i) = frame_orig(skip*i);
   }
}


void Qmf::CombinePoly(const gsl::vector &a_qmf1, const gsl::vector &a_qmf2,
               const double &qmf_gain, const int &Nsub, gsl::vector *a_combined) {
	// RADIX2 Implementation:
   size_t i;
   double temp;
   double thresh = 0.0000001;
	size_t Nlow = a_qmf1.size();
	size_t Nhigh = a_qmf2.size();
	int Ncomb = a_combined->size()-1;
	size_t nfft = NextPow2(Nsub);

	double *dataLow = (double *)calloc(nfft,sizeof(double)); /* complex, takes 2*nfft/2  values*/
	double *dataHigh = (double *)calloc(nfft,sizeof(double)); /* complex, takes 2*nfft/2  values*/
	double *dataComb = (double *)calloc(2*nfft,sizeof(double));
	gsl::vector fftlow(nfft/2);
	gsl::vector ffthigh(nfft/2);

	/** FFT LOW-BAND **/
	for (i=0; i<Nlow; i++)
		dataLow[i] = a_qmf1(i);

	gsl_fft_real_radix2_transform(dataLow, 1, nfft);

	fftlow(0) = 1.0/pow(dataLow[0],2);
	for(i=1; i<nfft/2; i++){
		temp = (pow(dataLow[i], 2) + pow(dataLow[nfft-i], 2));
		if(temp < thresh)
			temp = thresh;
		fftlow(i) = 1.0/temp;
	}

	double elow = 0.0;
	for(i = 0; i<fftlow.size(); i++) {
		elow += fftlow(i);
	}
	elow = sqrt(elow); // Low-band energy


	/** FFT HIGH-BAND **/
	for (i=0; i<Nhigh; i++)
		dataHigh[i] = a_qmf2(i);

	gsl_fft_real_radix2_transform(dataHigh, 1, nfft);

	ffthigh(0) = 1.0/pow(dataHigh[0],2);
	for(i=1; i<nfft/2; i++){
		temp = (pow(dataHigh[i], 2) + pow(dataHigh[nfft-i], 2));
		if(temp < thresh)
			temp = thresh;
		ffthigh(i) = 1.0/temp;
	}

	double ehigh = 0.0;
	for(i = 0; i<ffthigh.size(); i++) {
		ehigh += ffthigh(i);
	}
	ehigh = sqrt(ehigh); // High-band energy

	ffthigh.reverse(); // High-band has mirrored frequency

	/* Scale power of ffthigh according to QMF gain */
   ffthigh *= elow/ehigh * pow(10,qmf_gain/20.0);

	/** Combine PSD vectors **/
	for(i=0;i<nfft/2;i++)
		dataComb[i] = fftlow(i);

	for(i=nfft/2;i<nfft;i++)
		dataComb[i] = ffthigh(i-nfft/2);

	//dataComb[nfft] = dataComb[nfft-1];
	//for(i=nfft+1;i<2*nfft;i++)
	//	dataComb[i] = 0;

	/** IFFT combined PSD vector for Autocorrelation **/
	gsl_fft_halfcomplex_radix2_inverse(dataComb, 1,2*nfft);

	/** Set values to AC vector and compute minimum phase polynomial with Levinson **/
	gsl::vector ac(Ncomb+1);

	for(i=0;i<ac.size();i++)
    	ac(i) = dataComb[i];

   Levinson(ac, a_combined);

    // Free memory
	free(dataLow);
	free(dataHigh);
	free(dataComb);
}



