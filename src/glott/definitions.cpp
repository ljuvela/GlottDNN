/*
 * definitions.cpp
 *
 *  Created on: 30 Sep 2016
 *      Author: ljuvela
 */

#include "definitions.h"
#include <iostream>



Param::Param() {
	external_f0_filename = NULL;
	external_gci_filename = NULL;
	basename = NULL;
	default_windowing_function = HANN;
}

Param::~Param() {
	if (external_f0_filename)
		delete[] external_f0_filename;
	if (external_gci_filename)
		delete[] external_gci_filename;
	if (basename)
	   delete[] basename;
}

AnalysisData::AnalysisData() {}

AnalysisData::~AnalysisData() {}

int AnalysisData::AllocateData(const Param &params) {
	fundf = gsl::vector(params.number_of_frames,true);
	frame_energy = gsl::vector(params.number_of_frames,true);
	excitation_signal = gsl::vector(signal.size(), true);

	poly_vocal_tract = gsl::matrix(params.lpc_order_vt+1,params.number_of_frames,true);
	lsf_vocal_tract = gsl::matrix(params.lpc_order_vt,params.number_of_frames,true);
	poly_glott = gsl::matrix(params.lpc_order_glot+1,params.number_of_frames,true);
	lsf_glott = gsl::matrix(params.lpc_order_glot,params.number_of_frames,true);

	return EXIT_SUCCESS;
}

int AnalysisData::SaveData(const Param &params) {

   return EXIT_SUCCESS;
}
