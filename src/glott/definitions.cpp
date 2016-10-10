/*
 * definitions.cpp
 *
 *  Created on: 30 Sep 2016
 *      Author: ljuvela
 */
#include <iostream>
#include "FileIo.h"
#include "definitions.h"




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
	source_signal = gsl::vector(params.signal_length, true);

	poly_vocal_tract = gsl::matrix(params.lpc_order_vt+1,params.number_of_frames,true);
	lsf_vocal_tract = gsl::matrix(params.lpc_order_vt,params.number_of_frames,true);
	poly_glott = gsl::matrix(params.lpc_order_glot+1,params.number_of_frames,true);
	lsf_glott = gsl::matrix(params.lpc_order_glot,params.number_of_frames,true);

	excitation_pulses = gsl::matrix(params.paf_pulse_length, params.number_of_frames, true);

	return EXIT_SUCCESS;
}

int AnalysisData::SaveData(const Param &params) {

   if (params.extract_lsf_vt)
      WriteGslMatrix(params.basename, ".LSF", params.data_type, lsf_vocal_tract);
   if (params.extract_pulses_as_features)
      WriteGslMatrix(params.basename, ".PLS", params.data_type, excitation_pulses);
   if (params.extract_f0)
      WriteGslVector(params.basename, ".F0", params.data_type, fundf);
   // TODO: allow write only as .wav
   if (params.extract_glottal_excitation)
      WriteGslVector(params.basename, ".GlottalExcitation", params.data_type, source_signal);




   return EXIT_SUCCESS;
}
