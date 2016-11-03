/*
 * ReadConfig.cpp
 *
 *  Created on: 29 Sep 2016
 *      Author: ljuvela
 */

#include <iostream>
#include <libconfig.h++>
#include <cstring>
#include "definitions.h"
#include "ReadConfig.h"

int ConfigLookupInt(const char *config_string, const libconfig::Config &cfg, const bool default_config, int *val) {
	int ival;
	if (cfg.lookupValue(config_string, ival) == false) {
		if (default_config) {
			std::cerr << "Could not find value: " << config_string << " in default config" << std::endl ;
			return EXIT_FAILURE;
		}
	} else {
		*val = ival;
	}
	return EXIT_SUCCESS;
}

int ConfigLookupDouble(const char *config_string, const libconfig::Config &cfg, const bool default_config, double *val) {
	double dval;
	if (cfg.lookupValue(config_string, dval) == false) {
		if (default_config) {
			std::cerr << "Could not find value: " << config_string << " in default config" << std::endl ;
			return EXIT_FAILURE;
		}
	} else {
		*val = dval;
	}
	return EXIT_SUCCESS;
}

int ConfigLookupBool(const char *config_string, const libconfig::Config &cfg, const bool default_config, bool *val) {
	bool bval;
	if (cfg.lookupValue(config_string, bval) == false) {
		if (default_config) {
			std::cerr << "Could not find value: " << config_string << " in default config" << std::endl ;
			return EXIT_FAILURE;
		}
	} else {
		*val = bval;
	}
	return EXIT_SUCCESS;
}

int ConfigLookupCString(const char *config_string, const libconfig::Config &cfg, const bool default_config, char **val) {
	std::string sval;

	if (cfg.lookupValue(config_string, sval) == false) {
		if (default_config) {
			std::cerr << "Could not find value: " << config_string << " in default config" << std::endl ;
			return EXIT_FAILURE;
		}
	} else {
		*val = new char[sval.size()+1];
		std::strcpy(*val,sval.c_str());
	}
	return EXIT_SUCCESS;
}

int ConfigLookupString(const char *config_string, const libconfig::Config &cfg, const bool default_config, std::string &sval) {

   if (cfg.lookupValue(config_string, sval) == false) {
      if (default_config) {
         std::cerr << "Could not find value: " << config_string << " in default config" << std::endl ;
         return EXIT_FAILURE;
      }
   }
   return EXIT_SUCCESS;
}


int AssignConfigParams(const libconfig::Config &cfg, const bool default_config, Param *params) {

	if (ConfigLookupInt("SAMPLING_FREQUENCY", cfg, default_config, &(params->fs)) == EXIT_FAILURE)
		return EXIT_FAILURE;

   params->frame_length_long = (int)round(50/1000.0*(double)params->fs); // Hard coded value of 50ms.

   double shift_ms = -1.0;
   if (ConfigLookupDouble("FRAME_SHIFT", cfg, default_config, &(shift_ms)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if( default_config || shift_ms > 0)
      params->frame_shift = (int)round(shift_ms/1000.0*(double)params->fs);

   double frame_ms = -1.0;
   if (ConfigLookupDouble("FRAME_LENGTH", cfg, default_config, &(frame_ms)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if( default_config || frame_ms > 0)
      params->frame_length = (int)round(frame_ms/1000.0*(double)params->fs);

   frame_ms = -1.0;
   if (ConfigLookupDouble("UNVOICED_FRAME_LENGTH", cfg, default_config, &(frame_ms)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if( default_config || frame_ms > 0)
      params->frame_length_unvoiced = (int)round(frame_ms/1000.0*(double)params->fs);

	if (ConfigLookupBool("USE_EXTERNAL_F0", cfg, default_config, &(params->use_external_f0)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupString("EXTERNAL_F0_FILENAME", cfg, default_config, params->external_f0_filename) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupBool("USE_EXTERNAL_GCI", cfg, default_config, &(params->use_external_gci)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupString("EXTERNAL_GCI_FILENAME", cfg, default_config, params->external_gci_filename) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupDouble("GIF_PRE_EMPHASIS_COEFFICIENT", cfg, default_config, &(params->gif_pre_emphasis_coefficient)) == EXIT_FAILURE)
			return EXIT_FAILURE;

	if (ConfigLookupBool("USE_ITERATIVE_GIF", cfg, default_config, &(params->use_iterative_gif)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("LPC_ORDER_GLOT_IAIF", cfg, default_config, &(params->lpc_order_glot_iaif)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("LPC_ORDER_VT", cfg, default_config, &(params->lpc_order_vt)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("LPC_ORDER_GLOT", cfg, default_config, &(params->lpc_order_glot)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupDouble("WARPING_LAMBDA_VT", cfg, default_config, &(params->warping_lambda_vt)) == EXIT_FAILURE)
			return EXIT_FAILURE;

	if (ConfigLookupDouble("AME_DURATION_QUOTIENT", cfg, default_config, &(params->ame_duration_quotient)) == EXIT_FAILURE)
			return EXIT_FAILURE;

	if (ConfigLookupDouble("AME_POSITION_QUOTIENT", cfg, default_config, &(params->ame_position_quotient)) == EXIT_FAILURE)
			return EXIT_FAILURE;

	if (ConfigLookupBool("QMF_SUBBAND_ANALYSIS", cfg, default_config, &(params->qmf_subband_analysis)) == EXIT_FAILURE)
		return EXIT_FAILURE;

   if (ConfigLookupDouble("MAX_PULSE_LEN_DIFF", cfg, default_config, &(params->max_pulse_len_diff)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupInt("PAF_PULSE_LENGTH", cfg, default_config, &(params->paf_pulse_length)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_PULSE_INTERPOLATION", cfg, default_config, &(params->use_pulse_interpolation)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("HP_FILTERING", cfg, default_config, &(params->use_highpass_filtering)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_WAVEFORMS_DIRECTLY", cfg, default_config, &(params->use_waveforms_directly)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_F0", cfg, default_config, &(params->extract_f0)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_GAIN", cfg, default_config, &(params->extract_gain)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_LSF_VT", cfg, default_config, &(params->extract_lsf_vt)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_LSF_GLOT", cfg, default_config, &(params->extract_lsf_glot)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_HNR", cfg, default_config, &(params->extract_hnr)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_INFOFILE", cfg, default_config, &(params->extract_infofile)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_GLOTTAL_EXCITATION", cfg, default_config, &(params->extract_glottal_excitation)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_GCI_SIGNAL", cfg, default_config, &(params->extract_gci_signal)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_PULSES_AS_FEATURES", cfg, default_config, &(params->extract_pulses_as_features)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupInt("LPC_ORDER_QMF1", cfg, default_config, &(params->lpc_order_vt_qmf1)) == EXIT_FAILURE)
		return EXIT_FAILURE;

   if (ConfigLookupInt("LPC_ORDER_QMF2", cfg, default_config, &(params->lpc_order_vt_qmf2)) == EXIT_FAILURE)
		return EXIT_FAILURE;

   if (ConfigLookupDouble("F0_MIN", cfg, default_config, &(params->f0_min)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("F0_MAX", cfg, default_config, &(params->f0_max)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("VOICING_THRESHOLD", cfg, default_config, &(params->voicing_threshold)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("ZCR_THRESHOLD", cfg, default_config, &(params->zcr_threshold)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("RELATIVE_F0_THRESHOLD", cfg, default_config, &(params->relative_f0_threshold)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupInt("HNR_ORDER", cfg, default_config, &(params->hnr_order)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("SPEED_SCALE", cfg, default_config, &(params->speed_scale)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("PITCH_SCALE", cfg, default_config, &(params->pitch_scale)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("POSTFILTER_COEFFICIENT", cfg, default_config, &(params->postfilter_coefficient)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupBool("USE_POSTFILTERING", cfg, default_config, &(params->use_postfiltering)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_TRAJECTORY_SMOOTHING", cfg, default_config, &(params->use_trajectory_smoothing)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_SPECTRAL_MATCHING", cfg, default_config, &(params->use_spectral_matching)) == EXIT_FAILURE)
      return EXIT_FAILURE;

	if (ConfigLookupInt("LSF_VT_SMOOTH_LEN", cfg, default_config, &(params->lsf_vt_smooth_len)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("LSF_GLOT_SMOOTH_LEN", cfg, default_config, &(params->lsf_glot_smooth_len)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("GAIN_SMOOTH_LEN", cfg, default_config, &(params->gain_smooth_len)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("HNR_SMOOTH_LEN", cfg, default_config, &(params->hnr_smooth_len)) == EXIT_FAILURE)
		return EXIT_FAILURE;

   if (ConfigLookupDouble("NOISE_GAIN_VOICED", cfg, default_config, &(params->noise_gain_voiced)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupDouble("NOISE_LOW_FREQ_LIMIT_VOICED", cfg, default_config, &(params->noise_low_freq_limit_voiced)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupDouble("NOISE_GAIN_UNVOICED", cfg, default_config, &(params->noise_gain_unvoiced)) == EXIT_FAILURE)
      return EXIT_FAILURE;

	if (ConfigLookupString("DNN_WEIGHT_PATH", cfg, default_config, params->dnn_path_basename) == EXIT_FAILURE)
		return EXIT_FAILURE;

   double update_interval_ms=0;
	if (ConfigLookupDouble("FILTER_UPDATE_INTERVAL_VT", cfg, default_config, &(update_interval_ms)) == EXIT_FAILURE)
		return EXIT_FAILURE;
	if( default_config || update_interval_ms > 0)
	   params->filter_update_interval_vt = (int)round(update_interval_ms/1000.0*(double)params->fs);

	update_interval_ms=0;
	if (ConfigLookupDouble("FILTER_UPDATE_INTERVAL_SPECMATCH", cfg, default_config, &(update_interval_ms)) == EXIT_FAILURE)
		return EXIT_FAILURE;
	if( default_config || update_interval_ms > 0)
	    params->filter_update_interval_specmatch = (int)round(update_interval_ms/1000.0*(double)params->fs);

   if (ConfigLookupString("DATA_DIRECTORY", cfg, default_config, params->data_directory) == EXIT_FAILURE)
      return EXIT_FAILURE;

    if (ConfigLookupBool("USE_PITCH_SYNCHRONOUS_ANALYSIS", cfg, default_config, &(params->use_pitch_synchronous_analysis)) == EXIT_FAILURE)
      return EXIT_FAILURE;


   //if (ConfigLookupDouble("F0_CHECK_RANGE", cfg, default_config, &(params->f0_check_range)) == EXIT_FAILURE)
   //      return EXIT_FAILURE;

	/* Read enum style configurations */
	std::string str = "";

	/* Data Format */
	if (ConfigLookupString("DATA_TYPE", cfg, default_config, str) == EXIT_FAILURE)
	   return EXIT_FAILURE;
	if( default_config || str != "") {
	   if (str == "ASCII")
	      params->data_type = ASCII;
	   else if (str == "BINARY")
	      params->data_type = BINARY;
	   else
	      return EXIT_FAILURE;
	}


	/* Signal Polarity */
	str = "";
	if (ConfigLookupString("SIGNAL_POLARITY", cfg, default_config, str) == EXIT_FAILURE)
	   return EXIT_FAILURE;
	if( default_config || str != "") {
	   if (str == "DEFAULT")
	      params->signal_polarity = POLARITY_DEFAULT;
	   else if (str == "INVERT")
	      params->signal_polarity = POLARITY_INVERT;
	   else if (str == "DETECT")
	      params->signal_polarity = POLARITY_DETECT;
	   else
	      return EXIT_FAILURE;
	}



	/* LP weighting function */
	str = "";
	if (ConfigLookupString("LP_WEIGHTING_FUNCTION", cfg, default_config, str) == EXIT_FAILURE)
	   return EXIT_FAILURE;
	if( default_config || str != "") {
	   if (str == "NONE")
	      params->lp_weighting_function= NONE;
	   else if (str == "AME")
	      params->lp_weighting_function = AME;
	   else if (str == "STE")
	      params->lp_weighting_function = STE;
	   else
	      return EXIT_FAILURE;
	}



	/* Excitation method */
	str = "";
	if (ConfigLookupString("EXCITATION_METHOD", cfg, default_config, str) == EXIT_FAILURE)
	   return EXIT_FAILURE;
	if( default_config || str != "") {
	   if (str == "SINGLE_PULSE")
	      params->excitation_method = SINGLE_PULSE_EXCITATION;
	   else if (str == "DNN_GENERATED")
	      params->excitation_method = DNN_GENERATED_EXCITATION;
	   else if (str == "PULSES_AS_FEATURES")
	      params->excitation_method = PULSES_AS_FEATURES_EXCITATION;
	   else
	      return EXIT_FAILURE;
	}


	/* PSOLA window for synthesis */
	str = "";
	if (ConfigLookupString("PSOLA_WINDOW", cfg, default_config, str) == EXIT_FAILURE)
	   return EXIT_FAILURE;
	if( default_config || str != "") {
	   if (str == "NONE")
	      params->psola_windowing_function = RECT;
	   else if (str == "COSINE")
	      params->psola_windowing_function = COSINE;
	   else if (str == "HANN")
	      params->psola_windowing_function = HANN;
	   else
	      return EXIT_FAILURE;
	}





	return EXIT_SUCCESS;
}


int ReadConfig(const char *filename, const bool default_config, Param *params) {

   libconfig::Config cfg;
	/* Read the file. If there is an error, report it and exit. */
	try
	{
		cfg.readFile(filename);
	}
	catch(const libconfig::FileIOException &fioex)
	{
		std::cerr << "I/O error while reading file: " << filename << std::endl;
		return(EXIT_FAILURE);
	}
	catch(const libconfig::ParseException &pex)
	{
		std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
		            						  << " - " << pex.getError() << std::endl;
		return(EXIT_FAILURE);
	}

	AssignConfigParams(cfg, default_config, params);
	return EXIT_SUCCESS;
}

