/*
 * InverseFiltering.cpp
 *
 *  Created on: 3 Oct 2016
 *      Author: ljuvela
 */

#include <gslwrap/vector_double.h>
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
    *weight = gsl::vector(params.frame_length + params.lpc_order_vt);

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

/*
void WWLP(const gsl::vector &weight_function, const double &warping_lambda_vt,
		const int &lpc_order_vt, const gsl::vector &frame, gsl::vector *A) {




	int i,j,k,s,p = a->size-1;
	double win = 0.0,sum = 0.0;
	gsl::vector wframe(frame->size);
	gsl_vector *a_temp = gsl_vector_calloc(p);
	gsl_vector *r = gsl_vector_calloc(p+1);
	gsl_vector *b = gsl_vector_alloc(p);
	gsl_matrix *R = gsl_matrix_alloc (p, p);
	gsl_permutation *perm = gsl_permutation_alloc(p);



	// Copy warped frame
	gsl_vector *wframe_w = gsl_vector_calloc(wframe->size+p);

	for(i=0;i<wframe->size;i++)
		gsl_vector_set(wframe_w,i,gsl_vector_get(wframe,i));

	//for(i=0;i<wframe->size;i++)
	//	gsl_vector_set(wframe_w,i,gsl_vector_get(wframe,i));



	gsl_matrix *Y = gsl_matrix_calloc(p+1,wframe->size+p); // Delayed and weighted versions of the signal
	gsl_matrix *Rfull = gsl_matrix_calloc(p+1,p+1);

	// Matrix Dw
	for(j=0;j<wframe->size;j++) {
		if(Wn == NULL) {
			gsl_matrix_set(Y,0,j,gsl_vector_get(wframe,j));
		} else {
			gsl_matrix_set(Y,0,j,sqrt(gsl_vector_get(Wn,j))*gsl_vector_get(wframe,j));
		}

	}

	for(i=1;i<p+1;i++) {
		AllPassDelay2(wframe_w, lambda);
		for(j=0;j<wframe_w->size;j++) {
			if(Wn == NULL) {
				gsl_matrix_set(Y,i,j,gsl_vector_get(wframe_w,j));
			} else {
				gsl_matrix_set(Y,i,j,sqrt(gsl_vector_get(Wn,j))*gsl_vector_get(wframe_w,j));
			}
		}


	}



	//printf("%i, %i\n",Y->size1,Y->size2);

	// Rfull = Dw*Dw'
	gsl_blas_dgemm(CblasNoTrans,CblasTrans,1.0,Y,Y,0.0,Rfull);

	//MPrint1(Rfull);

	// Autocorrelation matrix R (R = (YT*Y)/N, size p*p) and vector b (size p)

	for(i=0;i<p;i++) {
		for(j=0;j<p;j++) {
			gsl_matrix_set(R,i,j,gsl_matrix_get(Rfull,i+1,j+1));
		}
		gsl_vector_set(b,i,gsl_matrix_get(Rfull,i+1,0));
		sum += gsl_vector_get(b,i);
	}


	//MPrint2(R);
	//VPrint1(b);

	//Ra=r solver (LU-decomposition) (Do not evaluate LU if sum = 0)

	if(sum != 0) {
		gsl_linalg_LU_decomp(R, perm, &s);
		gsl_linalg_LU_solve(R, perm, b, a_temp);
	}

	// Set LP-coefficients to vector "a"
	for(i=1; i<a->size; i++) {
		gsl_vector_set(a, i, (-1)*gsl_vector_get(a_temp, i-1));
	}
	gsl_vector_set(a, 0, 1);

	//VPrint2(a);
	// Stabilize unstable filter by scaling the poles along the unit circle
	AC_stabilize(a,frame->size);

	//Pole_stabilize(a);

	// Remove real roots
	if(ROOT_SCALING == 1) {
		if(a->size > ROOT_SCALE_MIN_DEGREE) {
			RealRootScale(a);
		}
	}

	// Replace NaN-values with zeros in case of all-zero frames
	for(i=0;i<a->size;i++) {
		if(gsl_isnan(gsl_vector_get(a,i)))
			gsl_vector_set(a,i,0);
	}

	// Free memory
	gsl_vector_free(wframe);
	gsl_vector_free(wframe_w);
	gsl_vector_free(a_temp);
	gsl_vector_free(r);
	gsl_vector_free(b);
	gsl_matrix_free(R);
	gsl_matrix_free(Rfull);
	gsl_matrix_free(Y);
	gsl_permutation_free(perm);


}*/

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
	//gsl::vector A = WWLP(weight_function, params.warping_lambda_vt, params.lpc_order_vt, &frame);
	LPC(frame_pre_emph, params.lpc_order_vt, A);

	if(use_iterative_gif) {
		std::cout << "TODO" << std::endl;
		// TODO
	}

}


