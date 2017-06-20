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

int ConfigLookupInt(const char *config_string, const libconfig::Config &cfg, const bool required, int *val) {
	int ival;
	if (cfg.lookupValue(config_string, ival) == false) {
		if (required) {
			std::cerr << "Could not find value: " << config_string << " in default config" << std::endl ;
			return EXIT_FAILURE;
		}
	} else {
		*val = ival;
	}
	return EXIT_SUCCESS;
}

int ConfigLookupDouble(const char *config_string, const libconfig::Config &cfg, const bool required, double *val) {
	double dval;
	if (cfg.lookupValue(config_string, dval) == false) {
		if (required) {
			std::cerr << "Could not find value: " << config_string << " in default config" << std::endl ;
			return EXIT_FAILURE;
		}
	} else {
		*val = dval;
	}
	return EXIT_SUCCESS;
}

int ConfigLookupBool(const char *config_string, const libconfig::Config &cfg, const bool required, bool *val) {
	bool bval;
	if (cfg.lookupValue(config_string, bval) == false) {
		if (required) {
			std::cerr << "Could not find value: " << config_string << " in default config" << std::endl ;
			return EXIT_FAILURE;
		}
	} else {
		*val = bval;
	}
	return EXIT_SUCCESS;
}

int ConfigLookupCString(const char *config_string, const libconfig::Config &cfg, const bool required, char **val) {
	std::string sval;

	if (cfg.lookupValue(config_string, sval) == false) {
		if (required) {
			std::cerr << "Could not find value: " << config_string << " in default config" << std::endl ;
			return EXIT_FAILURE;
		}
	} else {
		*val = new char[sval.size()+1];
		std::strcpy(*val,sval.c_str());
	}
	return EXIT_SUCCESS;
}

int ConfigLookupString(const char *config_string, const libconfig::Config &cfg, const bool required, std::string &sval) {

   if (cfg.lookupValue(config_string, sval) == false) {
      if (required) {
         std::cerr << "Could not find value: " << config_string << " in default config" << std::endl ;
         return EXIT_FAILURE;
      }
   }
   return EXIT_SUCCESS;
}


int AssignConfigParams(const libconfig::Config &cfg, const bool required, Param *params) {

	if (ConfigLookupInt("SAMPLING_FREQUENCY", cfg, required, &(params->fs)) == EXIT_FAILURE)
		return EXIT_FAILURE;

   params->frame_length_long = (int)round(50/1000.0*(double)params->fs); // Hard coded value of 50ms.

   double shift_ms = -1.0;
   if (ConfigLookupDouble("FRAME_SHIFT", cfg, required, &(shift_ms)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if( required || shift_ms > 0)
      params->frame_shift = (int)round(shift_ms/1000.0*(double)params->fs);

   double frame_ms = -1.0;
   if (ConfigLookupDouble("FRAME_LENGTH", cfg, required, &(frame_ms)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if( required || frame_ms > 0)
      params->frame_length = (int)round(frame_ms/1000.0*(double)params->fs);

   frame_ms = -1.0;
   if (ConfigLookupDouble("UNVOICED_FRAME_LENGTH", cfg, required, &(frame_ms)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if( required || frame_ms > 0)
      params->frame_length_unvoiced = (int)round(frame_ms/1000.0*(double)params->fs);

	if (ConfigLookupBool("USE_EXTERNAL_F0", cfg, required, &(params->use_external_f0)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupString("EXTERNAL_F0_FILENAME", cfg, required, params->external_f0_filename) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupBool("USE_EXTERNAL_GCI", cfg, required, &(params->use_external_gci)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupString("EXTERNAL_GCI_FILENAME", cfg, required, params->external_gci_filename) == EXIT_FAILURE)
		return EXIT_FAILURE;

   if (ConfigLookupBool("USE_EXTERNAL_LSF_VT", cfg, required, &(params->use_external_lsf_vt)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupString("EXTERNAL_LSF_VT_FILENAME", cfg, required, params->external_lsf_vt_filename) == EXIT_FAILURE)
      return EXIT_FAILURE;

	if (ConfigLookupDouble("GIF_PRE_EMPHASIS_COEFFICIENT", cfg, required, &(params->gif_pre_emphasis_coefficient)) == EXIT_FAILURE)
			return EXIT_FAILURE;

	if (ConfigLookupBool("USE_ITERATIVE_GIF", cfg, required, &(params->use_iterative_gif)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("LPC_ORDER_GLOT_IAIF", cfg, required, &(params->lpc_order_glot_iaif)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("LPC_ORDER_VT", cfg, required, &(params->lpc_order_vt)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("LPC_ORDER_GLOT", cfg, required, &(params->lpc_order_glot)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupDouble("WARPING_LAMBDA_VT", cfg, required, &(params->warping_lambda_vt)) == EXIT_FAILURE)
			return EXIT_FAILURE;

	if (ConfigLookupDouble("AME_DURATION_QUOTIENT", cfg, required, &(params->ame_duration_quotient)) == EXIT_FAILURE)
			return EXIT_FAILURE;

	if (ConfigLookupDouble("AME_POSITION_QUOTIENT", cfg, required, &(params->ame_position_quotient)) == EXIT_FAILURE)
			return EXIT_FAILURE;

	if (ConfigLookupBool("QMF_SUBBAND_ANALYSIS", cfg, required, &(params->qmf_subband_analysis)) == EXIT_FAILURE)
		return EXIT_FAILURE;

   if (ConfigLookupDouble("MAX_PULSE_LEN_DIFF", cfg, required, &(params->max_pulse_len_diff)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupInt("PAF_PULSE_LENGTH", cfg, required, &(params->paf_pulse_length)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_PULSE_INTERPOLATION", cfg, required, &(params->use_pulse_interpolation)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_WSOLA", cfg, false, &(params->use_wsola)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_PAF_UNVOICED", cfg, false, &(params->use_paf_unvoiced_synthesis)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_ORIGINAL_EXCITATION", cfg, false, &(params->use_original_excitation)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("HP_FILTERING", cfg, required, &(params->use_highpass_filtering)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_WAVEFORMS_DIRECTLY", cfg, required, &(params->use_waveforms_directly)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_F0", cfg, required, &(params->extract_f0)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_GAIN", cfg, required, &(params->extract_gain)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_LSF_VT", cfg, required, &(params->extract_lsf_vt)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_LSF_GLOT", cfg, required, &(params->extract_lsf_glot)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_HNR", cfg, required, &(params->extract_hnr)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_INFOFILE", cfg, required, &(params->extract_infofile)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_GLOTTAL_EXCITATION", cfg, required, &(params->extract_glottal_excitation)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_GCI_SIGNAL", cfg, required, &(params->extract_gci_signal)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("EXTRACT_PULSES_AS_FEATURES", cfg, required, &(params->extract_pulses_as_features)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupInt("LPC_ORDER_QMF1", cfg, required, &(params->lpc_order_vt_qmf1)) == EXIT_FAILURE)
		return EXIT_FAILURE;

   if (ConfigLookupInt("LPC_ORDER_QMF2", cfg, required, &(params->lpc_order_vt_qmf2)) == EXIT_FAILURE)
		return EXIT_FAILURE;

   if (ConfigLookupDouble("F0_MIN", cfg, required, &(params->f0_min)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("F0_MAX", cfg, required, &(params->f0_max)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("VOICING_THRESHOLD", cfg, required, &(params->voicing_threshold)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("ZCR_THRESHOLD", cfg, required, &(params->zcr_threshold)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("RELATIVE_F0_THRESHOLD", cfg, required, &(params->relative_f0_threshold)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupInt("HNR_ORDER", cfg, required, &(params->hnr_order)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("SPEED_SCALE", cfg, required, &(params->speed_scale)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("PITCH_SCALE", cfg, required, &(params->pitch_scale)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupDouble("POSTFILTER_COEFFICIENT", cfg, required, &(params->postfilter_coefficient)) == EXIT_FAILURE)
         return EXIT_FAILURE;
   
   if (ConfigLookupDouble("POSTFILTER_COEFFICIENT_GLOT", cfg, required, &(params->postfilter_coefficient_glot)) == EXIT_FAILURE)
         return EXIT_FAILURE;

   if (ConfigLookupBool("USE_POSTFILTERING", cfg, required, &(params->use_postfiltering)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_TRAJECTORY_SMOOTHING", cfg, required, &(params->use_trajectory_smoothing)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_GENERIC_ENVELOPE", cfg, required, &(params->use_generic_envelope)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   
   if (ConfigLookupBool("USE_SPECTRAL_MATCHING", cfg, required, &(params->use_spectral_matching)) == EXIT_FAILURE)
      return EXIT_FAILURE;

	if (ConfigLookupInt("LSF_VT_SMOOTH_LEN", cfg, required, &(params->lsf_vt_smooth_len)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("LSF_GLOT_SMOOTH_LEN", cfg, required, &(params->lsf_glot_smooth_len)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("GAIN_SMOOTH_LEN", cfg, required, &(params->gain_smooth_len)) == EXIT_FAILURE)
		return EXIT_FAILURE;

	if (ConfigLookupInt("HNR_SMOOTH_LEN", cfg, required, &(params->hnr_smooth_len)) == EXIT_FAILURE)
		return EXIT_FAILURE;

   if (ConfigLookupDouble("NOISE_GAIN_VOICED", cfg, required, &(params->noise_gain_voiced)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupDouble("NOISE_LOW_FREQ_LIMIT_VOICED", cfg, required, &(params->noise_low_freq_limit_voiced)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupDouble("NOISE_GAIN_UNVOICED", cfg, required, &(params->noise_gain_unvoiced)) == EXIT_FAILURE)
      return EXIT_FAILURE;

	if (ConfigLookupString("DNN_WEIGHT_PATH", cfg, required, params->dnn_path_basename) == EXIT_FAILURE)
		return EXIT_FAILURE;

   if (ConfigLookupBool("USE_PAF_ENERGY_NORM", cfg, required, &(params->use_paf_energy_normalization)) == EXIT_FAILURE)
      return EXIT_FAILURE;
   
  if (ConfigLookupBool("NOISE_GATED_SYNTHESIS", cfg, required, &(params->noise_gated_synthesis)) == EXIT_FAILURE)
      return EXIT_FAILURE;
  
     if (ConfigLookupDouble("NOISE_GATE_LIMIT_DB", cfg, required, &(params->noise_gate_limit_db)) == EXIT_FAILURE)
      return EXIT_FAILURE;
     
    if (ConfigLookupDouble("NOISE_REDUCTION_DB", cfg, required, &(params->noise_reduction_db)) == EXIT_FAILURE)
      return EXIT_FAILURE;
     

   double update_interval_ms=0;
	if (ConfigLookupDouble("FILTER_UPDATE_INTERVAL_VT", cfg, required, &(update_interval_ms)) == EXIT_FAILURE)
		return EXIT_FAILURE;
	if( required || update_interval_ms > 0)
	   params->filter_update_interval_vt = (int)round(update_interval_ms/1000.0*(double)params->fs);

	update_interval_ms=0;
	if (ConfigLookupDouble("FILTER_UPDATE_INTERVAL_SPECMATCH", cfg, required, &(update_interval_ms)) == EXIT_FAILURE)
		return EXIT_FAILURE;
	if( required || update_interval_ms > 0)
	    params->filter_update_interval_specmatch = (int)round(update_interval_ms/1000.0*(double)params->fs);

   if (ConfigLookupString("DATA_DIRECTORY", cfg, required, params->data_directory) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("SAVE_TO_DATADIR_ROOT", cfg, required, &(params->save_to_datadir_root)) == EXIT_FAILURE)
      return EXIT_FAILURE;

   if (ConfigLookupBool("USE_PITCH_SYNCHRONOUS_ANALYSIS", cfg, required, &(params->use_pitch_synchronous_analysis)) == EXIT_FAILURE)
     return EXIT_FAILURE;

   /* Lookup for parameter directory paths, always optional */
   ConfigLookupString("DIR_GAIN", cfg, false, params->dir_gain);
   ConfigLookupString("DIR_F0", cfg, false, params->dir_f0);
   ConfigLookupString("DIR_LSF_VT", cfg, false, params->dir_lsf);
   ConfigLookupString("DIR_LSF_GLOT", cfg, false, params->dir_lsfg);
   ConfigLookupString("DIR_HNR", cfg, false, params->dir_hnr);
   ConfigLookupString("DIR_PULSES_AS_FEATURES", cfg, false, params->dir_paf);
   ConfigLookupString("DIR_EXCITATION", cfg, false, params->dir_exc);
   ConfigLookupString("DIR_SPECTRUM", cfg, false, params->dir_sp);





	/* Read enum style configurations */
	std::string str;

	/* Data Format */
	str.clear();
	if (ConfigLookupString("DATA_TYPE", cfg, required, str) == EXIT_FAILURE)
	   return EXIT_FAILURE;
	if( required || str != "") {
	   if (str == "ASCII")
	      params->data_type = ASCII;
	   else if (str == "DOUBLE")
	      params->data_type = DOUBLE;
      else if (str == "FLOAT")
         params->data_type = FLOAT;
	   else {
	      std::cerr << "Error: DATA_TYPE must be either ASCII, DOUBLE of FLOAT" << std::endl;
	      return EXIT_FAILURE;
	   }
	}

	/* Signal Polarity */
	str.clear();
	if (ConfigLookupString("SIGNAL_POLARITY", cfg, required, str) == EXIT_FAILURE)
	   return EXIT_FAILURE;
	if( required || str != "") {
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
	str.clear();
	if (ConfigLookupString("LP_WEIGHTING_FUNCTION", cfg, required, str) == EXIT_FAILURE)
	   return EXIT_FAILURE;
	if( required || str != "") {
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
	str.clear();
	if (ConfigLookupString("EXCITATION_METHOD", cfg, required, str) == EXIT_FAILURE)
	   return EXIT_FAILURE;
	if( required || str != "") {
	   if (str == "SINGLE_PULSE")
	      params->excitation_method = SINGLE_PULSE_EXCITATION;
	   else if (str == "DNN_GENERATED")
	      params->excitation_method = DNN_GENERATED_EXCITATION;
	   else if (str == "PULSES_AS_FEATURES")
	      params->excitation_method = PULSES_AS_FEATURES_EXCITATION;
	   else {
	      std::cerr << "Error: invalid excitation method flag" << std::endl;
	      return EXIT_FAILURE;
	   }

	}

	/* PSOLA window for synthesis */
	str.clear();
	if (ConfigLookupString("PSOLA_WINDOW", cfg, required, str) == EXIT_FAILURE)
	   return EXIT_FAILURE;
	if( required || str != "") {
	   if (str == "NONE")
	      params->psola_windowing_function = RECT;
	   else if (str == "COSINE")
	      params->psola_windowing_function = COSINE;
	   else if (str == "HANN")
	      params->psola_windowing_function = HANN;
	   else
	      return EXIT_FAILURE;
	}

   /* Pulses-as-features analysis window */
   str.clear();
   if (ConfigLookupString("PAF_WINDOW", cfg, required, str) == EXIT_FAILURE)
      return EXIT_FAILURE;
   if( required || str != "") {
      if (str == "NONE")
         params->paf_analysis_window = RECT;
      else if (str == "COSINE")
         params->paf_analysis_window = COSINE;
      else if (str == "HANN")
         params->paf_analysis_window = HANN;
      else
         return EXIT_FAILURE;
   }


	return EXIT_SUCCESS;
}


int ReadConfig(const char *filename, const bool required, Param *params) {

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

	return AssignConfigParams(cfg, required, params);
	//return EXIT_SUCCESS;
}

