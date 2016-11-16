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
	dnn_path_basename = "";
	file_basename = "";
	data_directory = "";
	/* Enum parameters */
	default_windowing_function = NUTTALL;
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
	postfilter_coefficient = 0.4;
   use_trajectory_smoothing = false;
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
}

Param::~Param() {
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

   /* Write parameters to default directories if custom directories are not given */
   // TODO write one function/template to do this
   if (params.extract_gain) {
      if (params.dir_gain.empty()) {
         if (params.save_to_datadir_root)
            WriteGslVector(basedir + std::string(params.file_basename) + ".Gain", params.data_type, frame_energy);
         else
            WriteGslVector(basedir + "gain/" + std::string(params.file_basename) + ".Gain", params.data_type, frame_energy);
      } else {
         WriteGslVector(params.dir_gain + "/" + std::string(params.file_basename) + ".Gain", params.data_type, frame_energy);
      }
   }

   if (params.extract_lsf_vt){
      if (params.dir_lsf.empty()) {
         if (params.save_to_datadir_root)
            WriteGslMatrix(basedir + std::string(params.file_basename) + ".LSF", params.data_type, lsf_vocal_tract);
         else
            WriteGslMatrix(basedir + "lsf/" + std::string(params.file_basename) + ".LSF", params.data_type, lsf_vocal_tract);
      } else {
         WriteGslMatrix(params.dir_lsf + "/" + std::string(params.file_basename) + ".LSF", params.data_type, lsf_vocal_tract);
      }
   }

   if (params.extract_lsf_glot) {
      if (params.dir_lsfg.empty()) {
         if (params.save_to_datadir_root)
            WriteGslMatrix(basedir + std::string(params.file_basename) + ".LSFglot", params.data_type, lsf_glot);
         else
            WriteGslMatrix(basedir + "lsfg/" + std::string(params.file_basename) + ".LSFglot", params.data_type, lsf_glot);
      } else {
         WriteGslMatrix(params.dir_lsfg + "/" + std::string(params.file_basename) + ".LSFglot", params.data_type, lsf_glot);
      }
   }

   if (params.extract_hnr) {
      if (params.dir_hnr.empty()) {
         if (params.save_to_datadir_root)
            WriteGslMatrix(basedir + std::string(params.file_basename) + ".HNR", params.data_type, hnr_glot);
         else
            WriteGslMatrix(basedir + "hnr/" + std::string(params.file_basename) + ".HNR", params.data_type, hnr_glot);
      } else {
         WriteGslMatrix(params.dir_hnr + "/" + std::string(params.file_basename) + ".HNR", params.data_type, hnr_glot);
      }
   }

   if (params.extract_pulses_as_features) {
      if (params.dir_paf.empty()) {
         if (params.save_to_datadir_root)
            WriteGslMatrix(basedir + std::string(params.file_basename) + ".PAF", params.data_type, excitation_pulses);
         else
            WriteGslMatrix(basedir + "paf/" + std::string(params.file_basename) + ".PAF", params.data_type, excitation_pulses);
      } else {
         WriteGslMatrix(params.dir_hnr + "/"  + std::string(params.file_basename) + ".PAF", params.data_type, excitation_pulses);
      }
   }

   if (params.extract_f0) {
      if (params.dir_f0.empty()) {
         if (params.save_to_datadir_root)
            WriteGslVector(basedir + std::string(params.file_basename) + ".F0", params.data_type, fundf);
         else
            WriteGslVector(basedir + "f0/" + std::string(params.file_basename) + ".F0", params.data_type, fundf);
      } else {
         WriteGslVector(params.dir_f0 + "/" + std::string(params.file_basename) + ".F0", params.data_type, fundf);
      }
   }

   if (params.extract_glottal_excitation) {
      std::string exc_filename;
      if (params.dir_exc.empty()) {
         if (params.dir_exc.empty())
            exc_filename = basedir + std::string(params.file_basename) + ".src.wav";
         else
            exc_filename = basedir + "exc/" + std::string(params.file_basename) + ".src.wav";
      } else {
         exc_filename = params.dir_exc + "/" + std::string(params.file_basename) + ".src.wav";
      }
      if(WriteWavFile(exc_filename, source_signal, params.fs) == EXIT_FAILURE)
         return EXIT_FAILURE;
   }


   return EXIT_SUCCESS;
}

SynthesisData::SynthesisData() {}

SynthesisData::~SynthesisData() {}

