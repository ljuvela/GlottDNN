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
	external_f0_filename = "";
	external_gci_filename = "";
	external_lsf_vt_filename = "";
	dnn_path_basename = "";
	file_basename = "";
	data_directory = "";
	/* Enum parameters */
	default_windowing_function = HANN;
	signal_polarity = POLARITY_DEFAULT;
   lp_weighting_function = AME;
   excitation_method = SINGLE_PULSE_EXCITATION;
   psola_windowing_function = COSINE;
   paf_analysis_window = COSINE;
   /* Other parameters */
	fs = 16000;
	frame_length = 400;
	frame_length_unvoiced = 160;
	frame_length_long = 800;
	frame_shift = 80;
	number_of_frames = 0;
	signal_length = 0;
	lpc_order_vt = 30;
	lpc_order_glot = 10;
	hnr_order = 5;
	use_external_f0 = false;
	use_external_gci = false;
	use_external_lsf_vt = false;
	use_external_excitation = false;
	data_type = ASCII;
	qmf_subband_analysis = false;
	gif_pre_emphasis_coefficient = 0.97;
	unvoiced_pre_emphasis_coefficient = 0.00;
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
	use_paf_unvoiced_synthesis = false;
      use_velvet_unvoiced_paf = false;
	extract_f0 = true;
	extract_gain = true;
	extract_lsf_vt = true;
	extract_lsf_glot = true;
	extract_hnr = true;
	extract_infofile = false;
	extract_glottal_excitation = false;
	extract_gci_signal = false;
	extract_original_signal = false;
	extract_pulses_as_features = false;
	use_paf_energy_normalization = true;
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
   noise_gated_synthesis = false;
   noise_reduction_db = 20.0;
   noise_gate_limit_db = 80.0;
	postfilter_coefficient = 0.4;
   postfilter_coefficient_glot = 1.0;
   use_trajectory_smoothing = false;
   use_wsola = false;
   use_wsola_pitch_shift = false;
   lsf_vt_smooth_len = 3;
   lsf_glot_smooth_len = 5;
   gain_smooth_len = 5;
   hnr_smooth_len = 15;
   noise_gain_unvoiced = 1.0;
   noise_gain_voiced = 1.0;
   noise_low_freq_limit_voiced = 2500.0;
   filter_update_interval_vt = 80;
   filter_update_interval_specmatch = 80;
   use_pitch_synchronous_analysis = false;
   save_to_datadir_root = true;
   use_generic_envelope = false;
   extension_gain = ".gain";
   extension_lsf = ".lsf";
   extension_lsfg = ".slsf";
   extension_hnr = ".hnr";
   extension_paf = ".pls" ;
   extension_f0 = ".f0";
   extension_exc = ".exc.wav";
   extension_exc = ".src.wav";
   extension_syn = ".syn.wav";
   extension_wav = ".proc.wav";


}

Param::~Param() {
   // Replaced with std::string (automatic delete)
	/*if (external_f0_filename)
		delete[] external_f0_filename;
	if (external_gci_filename)
		delete[] external_gci_filename;
	if (basename)
	   delete[] basename;
   if (dnn_path_basename)
      delete[] dnn_path_basename;
   if (data_directory)
      delete[] data_directory;*/
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

   std::string basedir(params.data_directory) ;
   if (basedir.back() != '/')
      basedir += "/";

   std::string filename;
   if (params.extract_gain) {
      filename = GetParamPath("gain", params.extension_gain, params.dir_gain, params);
      WriteGslVector(filename, params.data_type, frame_energy);
   }
   if (params.extract_lsf_vt) {
      filename = GetParamPath("lsf", params.extension_lsf, params.dir_lsf, params);
      WriteGslMatrix(filename, params.data_type, lsf_vocal_tract);
   }
   if (params.extract_lsf_glot) {
      filename = GetParamPath("slsf", params.extension_lsfg, params.dir_lsfg, params);
      WriteGslMatrix(filename, params.data_type, lsf_glot);
   }
   if (params.extract_hnr) {
      filename = GetParamPath("hnr", params.extension_hnr, params.dir_hnr, params);
      WriteGslMatrix(filename, params.data_type, hnr_glot);
   }
   if (params.extract_pulses_as_features) {
      filename = GetParamPath("pls", params.extension_paf, params.dir_paf, params);
      WriteGslMatrix(filename, params.data_type, excitation_pulses);
   }
   if (params.extract_f0) {
      filename = GetParamPath("f0", params.extension_f0, params.dir_f0, params);
      WriteGslVector(filename, params.data_type, fundf);
   }
   if (params.extract_glottal_excitation) {
      filename = GetParamPath("exc", params.extension_src, params.dir_exc, params);
      if(WriteWavFile(filename, source_signal, params.fs) == EXIT_FAILURE)
         return EXIT_FAILURE;
   }
   if (params.extract_original_signal) {
      filename = GetParamPath("exc", params.extension_wav, params.dir_exc, params);
      if(WriteWavFile(filename, signal, params.fs) == EXIT_FAILURE)
         return EXIT_FAILURE;
   }

   return EXIT_SUCCESS;
}

SynthesisData::SynthesisData() {}

SynthesisData::~SynthesisData() {}

