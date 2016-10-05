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

void WWLP(const gsl::vector &weight_function, const double &warping_lambda_vt, const int &lpc_order_vt, const gsl::vector &frame, gsl::vector *A) {

	//

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
	//gsl::vector A = WWLP(weight_function, params.warping_lambda_vt, params.lpc_order_vt, &frame);
	LPC(frame_pre_emph, params.lpc_order_vt, A);

	if(use_iterative_gif) {
		std::cout << "TODO" << std::endl;
		// TODO
	}

}


