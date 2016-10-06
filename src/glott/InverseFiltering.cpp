/*
 * InverseFiltering.cpp
 *
 *  Created on: 3 Oct 2016
 *      Author: ljuvela
 */

#include <cmath>
#include <gslwrap/vector_double.h>
#include <gslwrap/matrix_double.h>
#include "definitions.h"
#include "SpFunctions.h"
#include "InverseFiltering.h"
#include "DebugUtils.h"

gsl::vector_int GetFrameGcis(const Param &params, const int frame_index, const gsl::vector_int &gci_inds) {
	/* Get frame sample range */
	int sample_index = round((double)params.signal_length*(double)frame_index/(double)params.number_of_frames);
	int minind = sample_index - round(params.frame_length/2);
	int maxind = sample_index + round(params.frame_length/2) - 1 ;
	size_t min_gci_ind, max_gci_ind;

	/* Find the range of gci inds */
	for (min_gci_ind=0; gci_inds(min_gci_ind) < minind; min_gci_ind++)
		if (min_gci_ind > gci_inds.size()-1) break;
	for (max_gci_ind=gci_inds.size()-1; gci_inds(max_gci_ind)>maxind; max_gci_ind--)
		if (max_gci_ind == 0) break;

	/* Allocate gci index vector */
	int n_gci_inds = max_gci_ind-min_gci_ind+1;
	gsl::vector_int frame_gci_inds;
	if (n_gci_inds > 0)
		frame_gci_inds = gsl::vector_int(n_gci_inds);

	/* Set frame gcis*/
	size_t i;
	for(i=0;i<frame_gci_inds.size();i++)
		frame_gci_inds(i) = gci_inds(min_gci_ind+i) - minind;

	return frame_gci_inds;
}

void LpWeightAme(const Param &params, const gsl::vector_int &gci_inds,
		 const size_t frame_index, gsl::vector *weight) {


	gsl::vector_int inds = GetFrameGcis(params, frame_index, gci_inds);

	if(!weight->is_set()) {
		*weight = gsl::vector(params.frame_length + params.lpc_order_vt);
	} else {
		if((int)weight->size() != params.frame_length + params.lpc_order_vt) {
			weight->resize(params.frame_length + params.lpc_order_vt);
		}
	}

   /* If unvoiced or GCIs not found, simply set weight to 1 */
	if(!inds.is_set()) {
		weight->set_all(1.0);
		return;
	}

	/* Algorithm parameters */
	double pq = params.ame_position_quotient;
	double dq = params.ame_duration_quotient;

	double d = 0.0000;
	//int nramp = DEFAULT_NRAMP;
	int nramp = 14;

	/* Sanity check */
	if(dq + pq > 1.0)
		dq = 1.0 - pq;

	/* Initialize */
	int i,j,t,t1 = 0,t2 = 0;

	/* Set weight according to GCIs */
	weight->set_all(d); // initialize to small value
	for(i=0;i<(int)inds.size()-1;i++) {
		t = inds(i+1)-inds(i);
		t1 = round(dq*t);
		t2 = round(pq*t);
		while(t1+t2 > t)
			t1 = t1-1;
		for(j=inds(i)+t2;j<inds(i)+t2+t1;j++)
			(*weight)(j) = 1.0;
		if(nramp > 0) {
			for(j=inds(i)+t2;j<inds(i)+t2+nramp;j++)
				(*weight)(j) = (j-inds(i)-t2+1)/(double)(nramp+1); // added double cast: ljuvela
			if(inds(i)+t2+t1-nramp >= 0)
				for(j=inds(i)+t2+t1-nramp;j<inds(i)+t2+t1;j++)
					(*weight)(j) = 1.0-(j-inds(i)-t2-t1+nramp+1)/(double)(nramp+1);
		}
	}
}

void LpWeightSte(const Param &params, const gsl::vector &frame, gsl::vector *weight) {
	if(!weight->is_set()) {
		*weight = gsl::vector(params.frame_length + params.lpc_order_vt);
	} else {
		if((int)weight->size() != params.frame_length + params.lpc_order_vt) {
			weight->resize(params.frame_length + params.lpc_order_vt);
		}
	}
	weight->set_all(1.0);
}

void GetLpWeight(const Param &params, const LpWeightingFunction &weight_type,
						const gsl::vector_int &gci_inds, const gsl::vector &frame,
						const size_t &frame_index, gsl::vector *weight_function) {

	switch(weight_type) {
	case NONE:
		weight_function->set_all(1.0);
		break;
	case AME:
		LpWeightAme(params, gci_inds, frame_index, weight_function);
		break;
	case STE:
		LpWeightSte(params, frame, weight_function);
		break;
	}
}



/**
 * Function WWLP
 *
 * Calculate Warped Weighted Linear Prediction (WWLP) coefficients using
 * autocorrelation method.
 *
 * @param frame pointer to the samples
 * @param a pointer to coefficiets (length p = LP degree)
 * @param Wn pointer to WWLP weighting function
 * @param lambda warping coefficient
 */
void WWLP(const gsl::vector &weight_function, const double &warping_lambda_vt, const LpWeightingFunction weight_type,
		const int &lp_order, const gsl::vector &frame, gsl::vector *A) {

   size_t i,j;
   size_t p = (size_t)lp_order;

  // Copy warped frame
   gsl::vector frame_w(frame.size()+p,true);
   for(i=0;i<frame.size();i++)
      frame_w(i) = frame(i);

   gsl::matrix Y(p+1,frame.size()+p,true); // Delayed and weighted versions of the signal

   // Matrix Dw
   for(j=0;j<frame.size();j++) { // Set first (unwarped) row
      if(weight_type == 0) {
         Y(0,j) = frame(j);
      } else {
         Y(0,j) = sqrt(weight_function(j))*frame(j);
      }
   }
   for(i=1;i<p+1;i++) { // Set delayed (warped) rows
      AllPassDelay(warping_lambda_vt, &frame_w);
      for(j=0;j<frame_w.size();j++) {
         if(weight_type == 0) {
            Y(i,j) = frame_w(j);
         } else {
            Y(i,j) = sqrt(weight_function(j))*frame_w(j);
         }
      }
   }
   // Rfull = Dw*Dw'
   gsl::matrix Rfull = Y*(Y.transpose());

   // Autocorrelation matrix R (R = (YT*Y)/N, size p*p) and vector b (size p)
   double sum = 0.0;
   gsl::matrix R(p,p);
   gsl::matrix b(p,1);
   for(i=0;i<p;i++) {
      for(j=0;j<p;j++) {
         R(i,j) = Rfull(i+1,j+1);
      }
      b(i,0) = Rfull(i+1,0);
      sum += b(i,0);
   }

   //Ra=b solver (LU-decomposition) (Do not evaluate LU if sum = 0)
   gsl::matrix a_tmp(p,1,true);
   if(sum != 0.0)
      a_tmp = R.LU_invert() * b;

	if(!A->is_set()) {
		*A = gsl::vector(p+1);
	} else {
		if(A->size() != p+1) {
			A->resize(p+1);
		} else {
			A->set_all(0.0);
		}
	}

   // Set LP-coefficients to vector "A"
   for(i=1; i<A->size(); i++) {
      (*A)(i) =  (-1.0)*a_tmp(i-1,0);
   }
   (*A)(0) = 1.0;

    // Stabilize unstable filter by scaling the poles along the unit circle
   //AC_stabilize(A,frame->size);
   //Pole_stabilize(a);


   // Replace NaN-values with zeros in case of all-zero frames
  // for(i=0;i<a->size;i++) {
   //   if(gsl_isnan((*A)(i)))
    //     (*A)(i) = (0.0);
   //}
}

void LPC(const gsl::vector &frame, const int &lpc_order, gsl::vector *A) {
	gsl::vector r;
	Autocorrelation(frame,lpc_order,&r);
	Levinson(r,A);
}

void ArAnalysis(const Param &params , const bool &use_iterative_gif,
					   const double &warping_lambda_vt, const gsl::vector &lp_weight,
					   const gsl::vector &frame, const gsl::vector &pre_frame, gsl::vector *A) {

	/* First-loop envelope */
	gsl::vector frame_pre_emph;
	Filter(std::vector<double>{1.0, params.gif_pre_emphasis_coefficient},std::vector<double>{1.0}, frame, &frame_pre_emph);
	ApplyWindowingFunction(params.default_windowing_function, &frame_pre_emph);
   WWLP(lp_weight, warping_lambda_vt, params.lp_weighting_function, params.lpc_order_vt, frame, A);
	//LPC(frame_pre_emph, params.lpc_order_vt, A);

	if(use_iterative_gif) {
		std::cout << "TODO" << std::endl;
		// TODO
	}

}


