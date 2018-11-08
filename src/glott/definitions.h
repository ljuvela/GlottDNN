// Copyright 2016-2018 Lauri Juvela and Manu Airaksinen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_

#include <gslwrap/vector_int.h>
#include <gslwrap/vector_double.h>
#include <gslwrap/matrix_double.h>

/* Enums */
enum DataType {ASCII, DOUBLE, FLOAT};
enum SignalPolarity {POLARITY_DEFAULT, POLARITY_INVERT, POLARITY_DETECT};
enum LpWeightingFunction {NONE, AME, STE};
enum WindowingFunctionType {HANN, HAMMING, BLACKMAN, COSINE, HANNING, RECT, NUTTALL};
enum ExcitationMethod {SINGLE_PULSE_EXCITATION, DNN_GENERATED_EXCITATION,
   PULSES_AS_FEATURES_EXCITATION, EXTERNAL_EXCITATION, IMPULSE_EXCITATION};

/* Structures */
struct Param
{
	Param();
	~Param();

  public:
	int fs;
	int frame_length;
	int frame_length_long;
	int frame_length_unvoiced;
	int frame_shift;
	int number_of_frames;
	int signal_length;
	int lpc_order_vt;
	int lpc_order_glot;
	int hnr_order;
	bool use_external_f0;
	std::string external_f0_filename;
	bool use_external_gci;
	std::string external_gci_filename;
	bool use_external_lsf_vt; // inverse filtering with external vocal tract filter
	std::string external_lsf_vt_filename;
	bool use_external_excitation;
	std::string external_excitation_filename;

	std::string dnn_path_basename;
	std::string data_directory;
	std::string file_basename;
	std::string file_path;
	bool save_to_datadir_root;
	DataType data_type;
	bool qmf_subband_analysis;
	SignalPolarity signal_polarity;
	double gif_pre_emphasis_coefficient;
	double unvoiced_pre_emphasis_coefficient;
	bool use_iterative_gif;
	LpWeightingFunction lp_weighting_function;
	int lpc_order_glot_iaif;
	double warping_lambda_vt;
	double ame_duration_quotient;
	double ame_position_quotient;
	WindowingFunctionType default_windowing_function;
	WindowingFunctionType psola_windowing_function;
	WindowingFunctionType paf_analysis_window;

	double max_pulse_len_diff;
	int paf_pulse_length;
	bool use_pulse_interpolation;
	bool use_highpass_filtering;
	bool use_waveforms_directly;
	bool use_paf_unvoiced_synthesis;
	bool use_velvet_unvoiced_paf;
	bool extract_f0;
	bool extract_gain;
	bool extract_lsf_vt;
	bool extract_lsf_glot;
	bool extract_hnr;
	bool extract_infofile;
	bool extract_glottal_excitation;
	bool extract_original_signal;
	bool extract_gci_signal;
	bool extract_pulses_as_features;
	bool use_paf_energy_normalization;
	int lpc_order_vt_qmf1;
	int lpc_order_vt_qmf2;
	double f0_max;
	double f0_min;
	double voicing_threshold;
	double zcr_threshold;
	double relative_f0_threshold;
	double speed_scale;
	double pitch_scale;
	bool use_postfiltering;
	bool use_spectral_matching;
	bool use_wsola;
	bool use_wsola_pitch_shift;
	bool noise_gated_synthesis;
	double noise_reduction_db;
	double noise_gate_limit_db;
	double postfilter_coefficient;
	double postfilter_coefficient_glot;
	bool use_trajectory_smoothing;
	int lsf_vt_smooth_len;
	int lsf_glot_smooth_len;
	int gain_smooth_len;
	int hnr_smooth_len;
	int filter_update_interval_vt;
	int filter_update_interval_specmatch;
	double noise_gain_unvoiced;
	double noise_gain_voiced;
	double noise_low_freq_limit_voiced;
	//double f0_check_range;
	ExcitationMethod excitation_method;
	bool use_pitch_synchronous_analysis;
	bool use_generic_envelope;

	/* directory paths for storing parameters */
	std::string dir_gain;
	std::string dir_lsf;
	std::string dir_lsfg;
	std::string dir_hnr;
	std::string dir_paf;
	std::string dir_f0;
	std::string dir_exc;
	std::string dir_syn;
	std::string dir_sp;

	/* extensions for parameter types */
	std::string extension_gain;
	std::string extension_lsf;
	std::string extension_lsfg;
	std::string extension_hnr;
	std::string extension_paf;
	std::string extension_f0;
	std::string extension_exc;
	std::string extension_src = ".src.wav";
	std::string extension_syn;
	std::string extension_wav;

	std::string wav_filename;
	std::string default_config_filename;
	std::string user_config_filename;
};

/* Define analysis data variable struct*/
struct AnalysisData {
	AnalysisData();
	~AnalysisData();
	int AllocateData(const Param &params);
	int SaveData(const Param &params);
public:
	gsl::vector signal;
	gsl::vector fundf;
	gsl::vector frame_energy;
	gsl::vector_int gci_inds;
	gsl::vector source_signal;
   gsl::vector source_signal_iaif;

	gsl::matrix poly_vocal_tract;
	gsl::matrix lsf_vocal_tract;
	gsl::matrix poly_glot;
	gsl::matrix lsf_glot;
	gsl::matrix excitation_pulses;
   gsl::matrix hnr_glot;


	/* QMF analysis specific */
	//gsl::matrix lsf_vt_qmf1;
	//gsl::matrix lsf_vt_qmf2;
	//gsl::vector gain_qmf;
};


/* Define analysis data variable struct*/
struct SynthesisData {
   SynthesisData();
   ~SynthesisData();
public:
   gsl::vector signal;
   gsl::vector fundf;
   gsl::vector frame_energy;
   gsl::vector excitation_signal;

   gsl::matrix poly_vocal_tract;
   gsl::matrix lsf_vocal_tract;
   gsl::matrix poly_glot;
   gsl::matrix lsf_glot;
   gsl::matrix excitation_pulses;
   gsl::matrix hnr_glot;
   
   gsl::matrix spectrum;
   
   /* QMF analysis specific */
   //gsl::matrix lsf_vt_qmf1;
   //gsl::matrix lsf_vt_qmf2;
   //gsl::vector gain_qmf;
};






#endif
