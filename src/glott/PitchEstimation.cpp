/*
 * PitchEstimation.cpp
 *
 *  Created on: 11 Oct 2016
 *      Author: ljuvela
 */



#include <gslwrap/vector_double.h>
#include <gslwrap/matrix_double.h>
#include <gslwrap/vector_int.h>

#include <gsl/gsl_fit.h>         /* GSL, Linear regression */
#include <gsl/gsl_multifit.h>    /* GSL, Higher order linear fitting */

#include "definitions.h"
#include "ComplexVector.h"
#include "SpFunctions.h"
#include "PitchEstimation.h"



/**
 * Function Parabolic_interpolation
 *
 * Fit quadratic function to the peak of autocorrelation function (ACF) to cancel the effect of the sampling period
 * (i.e. parabolic interpolation)
 *
 * @param r ACF
 * @param maxind index at maximum value
 * @param params
 * @return T0 value
 */
double Parabolic_interpolation(const gsl::vector &r, const int maxind) {

   /* Allocate variables */
   int i;
   double xi, chisq, T0;
   gsl_multifit_linear_workspace *work;
   gsl::matrix X(F0_INTERP_SAMPLES, 3);
   gsl::vector y(F0_INTERP_SAMPLES);
   gsl::vector w(F0_INTERP_SAMPLES);
   gsl::vector c(3);
   gsl::matrix cov(3, 3);
   work = gsl_multifit_linear_alloc(F0_INTERP_SAMPLES, 3);

   /* Set data */
   for(i=0;i<F0_INTERP_SAMPLES;i++) {
      xi = maxind-(F0_INTERP_SAMPLES-1)/2 + i;
      X(i,0) = 1.0;
      X(i,1) = xi;
      X(i,2) = xi*xi;
      y(i) = r(GSL_MIN(GSL_MAX(maxind-(F0_INTERP_SAMPLES-1)/2 + i,0),r.size()-1));
      w(i) = 1.0;
   }

   /* Quadratic fitting */
   /* Evaluate the value of the function at the zero of the derivative.
    * This is the interpolated length of the fundamental period in samples. */
   gsl_multifit_wlinear(X.gslobj(), w.gslobj(), y.gslobj(), c.gslobj(), cov.gslobj(), &chisq, work);
   T0 = -c(1)/(2.0*c(2));

   /* Free memory */
   gsl_multifit_linear_free(work);

   return T0;
}

void BandPassGain(const gsl::vector &frame, const int &fs, gsl::vector *bp_gain ) {

   /* FFT */
   ComplexVector c;
   FFTRadix2(frame, FFT_LENGTH, &c);
   gsl::vector fft = c.getAbs();

   /* Calculate gain for each band: 0--1, 1--2, 2--4, 4--6, 6--FS/2 kHz */
   int freq_lims[6] = {0,
         GSL_MIN(1000,fs/2),
         GSL_MIN(2000,fs/2),
         GSL_MIN(4000,fs/2),
         GSL_MIN(6000,fs/2),
         fs/2};
   double weights[5] = {1,1,0.5,0.5,1000/(double)(fs/2-freq_lims[4])};
   int i,k;
   int K = 5;
   *bp_gain = gsl::vector(K);
   for(k=0; k<K; k++) {
      int start_freq = rint(freq_lims[k]/(double)fs*(double)FFT_LENGTH);
      int stop_freq = rint(freq_lims[k+1]/(double)fs*(double)FFT_LENGTH);
      for(i=start_freq; i<stop_freq; i++)
         (*bp_gain)(k) += weights[k]*fft(i)*fft(i);
   }
}

void FundamentalFrequency(const Param &params, const gsl::vector &glottal_frame,
      const gsl::vector &signal_frame, double *fundf, gsl::vector *fundf_candidates) {

   // glottal_frame: is long iaif frame
   // signal_frame: short speech frame

   size_t i,n;
   double r_max;
   gsl::vector_int max_inds(NUMBER_OF_F0_CANDIDATES);
   gsl::vector_int max_inds_interp(NUMBER_OF_F0_CANDIDATES);


   /* Count the number of zero crossings */
   int zero_crossings = 0;
   for(i=0; i<signal_frame.size()-1; i++)
      if(signal_frame(i) * signal_frame(i+1) < 0)
         zero_crossings++;

   /* Calculate band energies */
   gsl::vector bp_gain;
   BandPassGain(signal_frame, params.fs, &bp_gain);

   /* Autocorrelation sequence */
   // TODO: fast autocorrelation
   gsl::vector r(glottal_frame.size());
   for(i=0; i<glottal_frame.size(); i++)
      for(n=i; n<glottal_frame.size(); n++)
         r(i)+= glottal_frame(n)*glottal_frame(n-i);

   /* Normalize r */
   r /= r.max();

   /* Copy vector r for interpolation */
   gsl::vector r_copy(r);

   /* Clear samples when the index exceeds the fundamental frequency limits */
   for(i=0; i<(size_t)rint(params.fs/params.f0_max); i++)
      r(i) = 0.0;
   for(i=rint(params.fs/params.f0_min); i<r.size(); i++)
      r(i) = 0.0;

   /* Clear samples descending from the end-points */
   size_t ind = rint(params.fs/params.f0_max);
   while(r(ind)-r(ind+1) > 0) {
      r(ind) = 0.0;
      ind++;
      if(ind+1>r.size()-1)
         break;
   }
   ind = rint(params.fs/params.f0_min)-1;
   while(r(ind)-r(ind-1) > 0) {
      r(ind) = 0.0;
      ind--;
      if(ind<1) {
         break;
      }
   }

   /* Get T0 and F0 candidates */
   for(i=0;i<NUMBER_OF_F0_CANDIDATES;i++) {

      /* Get the i:th T0 index estimate */
      max_inds(i) =  r.max_index();

      /* Fit quadratic function to the peak of ACF to cancel the effect of the sampling period
       * (i.e. parabolic interpolation) */
      double T0 = Parabolic_interpolation(r,max_inds(i));
      if(!isnan(T0) && params.fs/T0 < params.f0_max && params.fs/T0 > params.f0_min)
         max_inds_interp(i) = T0;
      else
         max_inds_interp(i) = 0.0;

      /* Set the F0 candidate */
      if(max_inds_interp(i) <= 0) {
         (*fundf_candidates)(i) = 0.0;
         break;
      } else {
         (*fundf_candidates)(i) = params.fs/max_inds_interp(i);
            if((*fundf_candidates)(i) > params.f0_max || (*fundf_candidates)(i) < params.f0_min
                  || bp_gain(0) < params.voicing_threshold/2.0 || zero_crossings > params.zcr_threshold*2.0)
               (*fundf_candidates)(i) = 0.0;
      }
      // TODO: use relative energy/sample threshold instead of absolute energy/frame! (ljuvela )


      /* Clear the descending samples from the i:th maximum */
      ind = max_inds(i);
      while(r(ind)-r(ind+1) > 0) {
         r(ind) = 0.0;
         ind++;
         if(ind+1>r.size()-1) {
            break;
         }
      }
      ind = GSL_MAX(max_inds(i)-1,1);
      while(r(ind)-r(ind-1) > 0) {
         r(ind) = 0.0;
         ind--;
         if(ind-1<0) {
            break;
         }
      }
   }

   /* Decide voiced/unvoiced. If voiced, set F0, otherwise set zero */
   if(bp_gain(0) < params.voicing_threshold
         || zero_crossings > params.zcr_threshold || max_inds_interp(0) == 0)
      *fundf =  0.0;
   else
      *fundf = params.fs/(double)max_inds_interp(0);



}


