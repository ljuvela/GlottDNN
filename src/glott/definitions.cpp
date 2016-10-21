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
   /* String parameters */
	external_f0_filename = NULL;
	external_gci_filename = NULL;
	dnn_path_basename = NULL;
	basename = NULL;
	/* Enum parameters */
	default_windowing_function = HANN;
	signal_polarity = POLARITY_DEFAULT;
   lp_weighting_function = AME;
   excitation_method = SINGLE_PULSE_EXCITATION;
   psola_windowing_function = COSINE;
   /* Other parameters */
	fs = 16000;
	frame_length = 400;
	frame_length_long = 800;
	frame_shift = 80;
	number_of_frames = 0 ;
	signal_length = 0;
	lpc_order_vt = 30;
	lpc_order_glot = 10;
	hnr_order = 5;
	use_external_f0 = false;
	use_external_gci = false;
	data_type = ASCII;
	qmf_subband_analysis = false;
	gif_pre_emphasis_coefficient = 0.97;
	use_iterative_gif = false;
	lpc_order_glot_iaif = 8;
	warping_lambda_vt = 0.0;
	ame_duration_quotient = 0.7;
	ame_position_quotient = 0.1;
	max_pulse_len_diff = 0.10;
	paf_pulse_length = 400;
	use_pulse_interpolation = true;
	use_highpass_filtering = true;
	use_waveforms_directly = false;
	extract_f0 = true;
	extract_gain = true;
	extract_lsf_vt = true;
	extract_lsf_glot = true;
	extract_hnr = true;
	extract_infofile = false;
	extract_glottal_excitation = false;
	extract_gci_signal = false;
	extract_pulses_as_features = false;
	lpc_order_vt_qmf1 = 48;
	lpc_order_vt_qmf2 = 12;
	f0_max = 500;
	f0_min = 50;
	voicing_threshold = 60.0;
	zcr_threshold = 120.0;
	relative_f0_threshold = 0.005;
	speed_scale = 1.0;
	pitch_scale = 1.0;
	use_postfiltering = false;
	use_spectral_matching = true;
	postfilter_coefficient = 0.4;
   use_trajectory_smoothing = false;
   lsf_vt_smooth_len = 3;
   lsf_glot_smooth_len = 5;
   gain_smooth_len = 5;
   hnr_smooth_len = 15;
   noise_gain_unvoiced = 1.0;
   noise_gain_voiced = 1.0;
   noise_low_freq_limit_voiced = 2500.0;

}

Param::~Param() {
	if (external_f0_filename)
		delete[] external_f0_filename;
	if (external_gci_filename)
		delete[] external_gci_filename;
	if (basename)
	   delete[] basename;
   if (dnn_path_basename)
      delete[] dnn_path_basename;
}

AnalysisData::AnalysisData() {}

AnalysisData::~AnalysisData() {}

int AnalysisData::AllocateData(const Param &params) {
	fundf = gsl::vector(params.number_of_frames,true);
	frame_energy = gsl::vector(params.number_of_frames,true);
	source_signal = gsl::vector(params.signal_length, true);

	poly_vocal_tract = gsl::matrix(params.lpc_order_vt+1,params.number_of_frames,true);
	lsf_vocal_tract = gsl::matrix(params.lpc_order_vt,params.number_of_frames,true);
	poly_glot = gsl::matrix(params.lpc_order_glot+1,params.number_of_frames,true);
	lsf_glot = gsl::matrix(params.lpc_order_glot,params.number_of_frames,true);
   hnr_glot = gsl::matrix(params.hnr_order,params.number_of_frames,true);

	excitation_pulses = gsl::matrix(params.paf_pulse_length, params.number_of_frames, true);

	//if(params.qmf_subband_analysis) {
   //lsf_vt_qmf1 = gsl::matrix(params.lpc_order_vt_qmf1+1,params.number_of_frames, true); // Includes QMF gain
   //   lsf_vt_qmf2 = gsl::matrix(params.lpc_order_vt_qmf2,params.number_of_frames, true);
   //   gain_qmf = gsl::vector(params.number_of_frames,true);
	//}

	return EXIT_SUCCESS;
}

int AnalysisData::SaveData(const Param &params) {

   if (params.extract_gain)
      WriteGslVector(params.basename, ".Gain", params.data_type, frame_energy);
   if (params.extract_lsf_vt)
      WriteGslMatrix(params.basename, ".LSF", params.data_type, lsf_vocal_tract);
   if (params.extract_lsf_glot)
      WriteGslMatrix(params.basename, ".LSFsource", params.data_type, lsf_glot);
   if (params.extract_hnr)
   if (params.extract_hnr)
      WriteGslMatrix(params.basename, ".HNR", params.data_type, hnr_glot);
   if (params.extract_pulses_as_features)
      WriteGslMatrix(params.basename, ".PLS", params.data_type, excitation_pulses);
   if (params.extract_f0)
      WriteGslVector(params.basename, ".F0", params.data_type, fundf);
   // TODO: allow write only as .wav
   if (params.extract_glottal_excitation)
      WriteGslVector(params.basename, ".GlottalExcitation", params.data_type, source_signal);

   return EXIT_SUCCESS;
}

SynthesisData::SynthesisData() {}

SynthesisData::~SynthesisData() {}

