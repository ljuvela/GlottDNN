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
 * Function ParabolicInterpolation
 *
 * Fit quadratic function to the peak of autocorrelation function (ACF) to cancel the effect of the sampling period
 * (i.e. parabolic interpolation)
 *
 * @param r ACF
 * @param maxind index at maximum value
 * @param params
 * @return T0 value
 */
double ParabolicInterpolation(const gsl::vector &r, const int maxind) {

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
      y(i) = r(GSL_MIN(GSL_MAX(maxind-(F0_INTERP_SAMPLES-1)/2 + i,0), (int)r.size()-1));
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


/**
 * Fill_f0_gaps
 *
 * Fill small gaps in F0
 *
 * @param fundf F0
 */
void FillF0Gaps(gsl::vector *fundf_ptr) {

   gsl::vector & fundf = *fundf_ptr;

   // constant hat trick values
   const int F0_FILL_RANGE = 6;
   const double RELATIVE_F0_THRESHOLD = 0.005;

   int i,j,voiced;
   double fundf_est,mu,std,sum,n,lim,ave;
   double f0jump01,f0jump45,f0jump02,f0jump03,f0jump12,f0jump13,f0jump42,f0jump43,f0jump52,f0jump53,f0jump05,f0jump14;
   gsl::vector fundf_fill(F0_FILL_RANGE);

   /* Estimate mean (mu) and standard deviation (std) of voiced parts */
   sum = 0;
   n = 0;
   for(i=0;i<(int)fundf.size();i++) {
      if(fundf(i) > 0) {
         sum += fundf(i);
         n++;
      }
   }
   mu = sum/n;
   sum = 0;
   n = 0;
   for(i=0;i<(int)fundf.size();i++) {
      if(fundf(i) > 0) {
         sum += pow(mu-fundf(i),2);
         n++;
      }
   }
   std = sqrt(sum/(n-1));

   /* Go through all F0 values and fill small gaps (even if voiced) */
   for(i=0;i<(int)fundf.size()-F0_FILL_RANGE;i++) {
      fundf_est = 0;
      voiced = 0;
      for(j=0;j<F0_FILL_RANGE;j++) {
         fundf_fill(j) = fundf(i+j);
         if(fundf_fill(j) > 0) {
            voiced++;
            fundf_est += fundf_fill(j);
         }
      }
      if(fundf_fill(0) > 0 && fundf_fill(1) > 0 && fundf_fill(5) > 0 && fundf_fill(4) > 0) {
         if(fundf_fill(2) == 0 && fundf_fill(3) == 0) {
            fundf(i+2) = fundf_est/4.0;
            fundf(i+3) = fundf_est/4.0;
         } else if (fundf_fill(2) == 0) {
            fundf(i+2) = fundf_est/5.0;
         } else if (fundf_fill(3) == 0) {
            fundf(i+3) = fundf_est/5.0;
         }
      }

      /* If all values are voiced, replace gaps of size two of which values differ significantly from average value */
      if(voiced == F0_FILL_RANGE) {
         f0jump01 = fabs(fundf_fill(0)-fundf_fill(1));
         f0jump45 = fabs(fundf_fill(4)-fundf_fill(5));
         f0jump02 = fabs(fundf_fill(0)-fundf_fill(2));
         f0jump03 = fabs(fundf_fill(0)-fundf_fill(3));
         f0jump12 = fabs(fundf_fill(1)-fundf_fill(2));
         f0jump13 = fabs(fundf_fill(1)-fundf_fill(3));
         f0jump42 = fabs(fundf_fill(4)-fundf_fill(2));
         f0jump43 = fabs(fundf_fill(4)-fundf_fill(3));
         f0jump52 = fabs(fundf_fill(5)-fundf_fill(2));
         f0jump53 = fabs(fundf_fill(5)-fundf_fill(3));
         f0jump05 = fabs(fundf_fill(0)-fundf_fill(5));
         f0jump14 = fabs(fundf_fill(1)-fundf_fill(4));
         lim = RELATIVE_F0_THRESHOLD*std;
         ave = (fundf_fill(0) + fundf_fill(1) + fundf_fill(4) + fundf_fill(5))/4.0;
         if(f0jump01 < lim && f0jump45 < lim &&
               f0jump02 > lim && f0jump03 > lim &&
               f0jump12 > lim && f0jump13 > lim &&
               f0jump42 > lim && f0jump43 > lim &&
               f0jump52 > lim && f0jump53 > lim &&
               f0jump05 < lim && f0jump14 < lim) {
            fundf(i+2) = ave;
            fundf(i+3) = ave;
         }
      }
   }

   /* Go through all F0 values and eliminate small voiced regions */
   for(i=0;i<(int)fundf.size()-F0_FILL_RANGE;i++) {
      for(j=0;j<F0_FILL_RANGE;j++) {
         fundf_fill(j) = fundf(i+j);
      }
      if(fundf_fill(0) == 0 && fundf_fill(1) == 0 && fundf_fill(5) == 0 && fundf_fill(4) == 0) {
         if(fundf_fill(2) > 0 || fundf_fill(3) > 0) {
            fundf(i+2) = 0;
            fundf(i+3) = 0;
         }
      }
   }
}


/**
 * Fundf_postprocessing
 *
 * Refine f0-trajectory
 *
 * @param fundf fundamental frequency values
 * @param fundf_orig original fundamental frequency values
 * @param fundf_candidates candidates for F0
 * @param params parameter structure
 */
void FundfPostProcessing(const Param &params, const gsl::vector &fundf_orig, const gsl::matrix &fundf_candidates, gsl::vector *fundf_ptr) {

   gsl::vector & fundf = *fundf_ptr;

   const double RELATIVE_F0_THRESHOLD = 0.005;
   const int F0_CHECK_RANGE = 10;

   bool f0_postprocessing = true;

   int i,j,ind,voiced_ind_b,voiced_ind_f,unvoiced_ind,n;
   double f0_forward,f0_backward,x[F0_CHECK_RANGE],y[F0_CHECK_RANGE],w[F0_CHECK_RANGE],f0jump_b,f0jump_f;
   double c0,c1,cov00,cov01,cov11,chisq_f,chisq_b,mu,std,sum;

   /* Estimate mean (mu) and standard deviation (std) of voiced parts */
   sum = 0;
   n = 0;
   for(i=0;i<(int)fundf.size();i++) {
      if(fundf(i) > 0) {
         sum += fundf(i);
         n++;
      }
   }
   mu = sum/n;
   sum = 0;
   n = 0;
   for(i=0;i<(int)fundf.size();i++) {
      if(fundf(i) > 0) {
         sum += pow(mu-fundf(i),2);
         n++;
      }
   }
   std = sqrt(sum/(n-1));

   /* Go through all F0 values and create weighted linear estimates for all F0 samples both from backward and forward */
   // TODO: Nonlinear fitting! (replace linear fit with spline interpolation?)
   if(f0_postprocessing) {

      /* Start looping F0 */
      for(i=F0_CHECK_RANGE;i<(int)fundf.size()-F0_CHECK_RANGE;i++) {

         /* Check values backward (left to right) */
         ind = 0;
         voiced_ind_b = 0;
         unvoiced_ind = 0;
         for(j=F0_CHECK_RANGE;j>0;j--) {
            if(fundf(i-j) == 0) {
               w[ind] = 0; // zero weight for unvoiced
               unvoiced_ind++;
            } else {
               w[ind] = 1; // unit weight for voiced
               voiced_ind_b++;
            }
            x[ind] = ind;
            y[ind] = fundf(i-j);
            ind++;
         }
         if(unvoiced_ind == F0_CHECK_RANGE || voiced_ind_b < 3) {
            f0_backward = 0;
         } else {

            /* Weighted linear fitting, estimate new value */
            gsl_fit_wlinear(x,1,w,1,y,1,(double)F0_CHECK_RANGE,&c0,&c1,&cov00,&cov01,&cov11,&chisq_b);
            f0_backward = c0 + c1*(double)F0_CHECK_RANGE;
         }

         /* Check values forward (right to left) */
         ind = 0;
         voiced_ind_f = 0;
         unvoiced_ind = 0;
         for(j=1;j<F0_CHECK_RANGE+1;j++) {
            if(fundf(i+j) == 0) {
               w[ind] = 0;
               unvoiced_ind++;
            } else {
               w[ind] = 1;
               voiced_ind_f++;
            }
            x[ind] = ind;
            y[ind] = fundf(i+j);
            ind++;
         }
         if(unvoiced_ind == F0_CHECK_RANGE || voiced_ind_f < 3) {
            f0_forward = 0;
         } else {

            /* Weighted linear fitting, estimate new value */
            gsl_fit_wlinear(x,1,w,1,y,1,(double)F0_CHECK_RANGE,&c0,&c1,&cov00,&cov01,&cov11,&chisq_f);
            f0_forward = c0 - c1;
         }

         /* Evaluate relative jump in F0 from both directions */
         if(fundf(i) != 0) {
            f0jump_b = fabs(fundf(i)-fundf(i-1))/fundf(i);
            f0jump_f = fabs(fundf(i)-fundf(i+1))/fundf(i);
         } else {
            f0jump_b = 0;
            f0jump_f = 0;
         }

         /* Set estimate if the relative jump is high enough,
          * take into account the number of used voiced values
          * in the fitting (voiced_ind) and the magnitude of the fit residual (chisq) */
         if(fundf(i) != 0) {
            double new_f0 = fundf(i);
            double f0jump = 0;
            if(voiced_ind_b > voiced_ind_f) { // Compare number of voiced frames
               if(f0_backward > params.f0_min && f0_backward < params.f0_max && f0jump_b >= RELATIVE_F0_THRESHOLD) {
                  new_f0 = f0_backward;
                  f0jump = f0jump_b;
               }
            } else if(voiced_ind_b < voiced_ind_f) {
               if(f0_forward > params.f0_min && f0_forward < params.f0_max && f0jump_f >= RELATIVE_F0_THRESHOLD) {
                  new_f0 = f0_forward;
                  f0jump = f0jump_f;
               }
            } else {
               if(chisq_b < chisq_f) { // Compare goodness of fit
                  if(f0_backward > params.f0_min && f0_backward < params.f0_max && f0jump_b >= RELATIVE_F0_THRESHOLD) {
                     new_f0 = f0_backward;
                     f0jump = f0jump_b;
                  }
               } else {
                  if(f0_forward > params.f0_min && f0_forward < params.f0_max && f0jump_f >= RELATIVE_F0_THRESHOLD) {
                     new_f0 = f0_forward;
                     f0jump = f0jump_f;
                  }
               }
            }

            /* Check if second F0 candidate is close. If so, set that value instead if estimated value */
            if(f0jump > RELATIVE_F0_THRESHOLD*std) {
               if(fabs(fundf_candidates(i,1) - new_f0) < f0jump)
                  fundf(i) = fundf_candidates(i,1);
               else
                  fundf(i) = new_f0;
            }
         }

         /* Make sure the correction does not lead the F0 trajectory to a wrong track by estimating
          * the cumulative difference between the original and new F0 curves.
          * This is just a precaution, leading to a wrong F0 track should not normally happen */
         double F0_CUM_ERROR_LIM = 7.0;
         int F0_CUM_ERROR_RANGE = 30;
         double cum_error = 0;
         for(j=GSL_MAX(i-F0_CUM_ERROR_RANGE,0);j<i;j++)
            cum_error += fabs(fundf_orig(j)-fundf(j))/mu;
         if(cum_error > F0_CUM_ERROR_LIM)
            fundf(i) = fundf_orig(i);
      }
   }
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
   //double r_max;
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
   gsl::vector r(glottal_frame.size(),true);
   gsl::vector ac;
   FastAutocorr(glottal_frame, &ac);
 
   /* Copy relevant indeces of ac to r (according to f0_max and f0_min) */
   for(i=(size_t)rint(params.fs/params.f0_max); i<GSL_MIN(rint(params.fs/params.f0_min)+1,glottal_frame.size()); i++)
          r(i) = ac(ac.size()/2+1+i); // floor(ac.size/2)+1

   /* Clear samples descending from the end-points */
   int ind = rint(params.fs/params.f0_max);
   while(r(ind)-r(ind+1) > 0) {
      r(ind) = 0.0;
      ind++;
      if(ind+1>(int)r.size()-1)
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
      double T0 = ParabolicInterpolation(r,max_inds(i));
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
         if(ind+1>(int)r.size()-1) {
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


