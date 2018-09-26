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

#include <iostream>
#include <exception>
#include <libconfig.h++>
#include <cstring>
#include "definitions.h"
#include "ReadConfig.h"

struct LookupException : public std::exception {
  const char *what() const throw() { return "Exiting due to missing config"; }
};

int ConfigLookupInt(const char *config_string, const libconfig::Config &cfg,
                    const bool required, int *val) {
  int ival;
  if (cfg.lookupValue(config_string, ival) == false) {
    if (required) {
      std::cerr << "Could not find value: " << config_string
                << " in default config" << std::endl;
      throw LookupException();
      return EXIT_FAILURE;
    }
  } else {
    *val = ival;
  }
  return EXIT_SUCCESS;
}

int ConfigLookupDouble(const char *config_string, const libconfig::Config &cfg,
                       const bool required, double *val) {
  double dval;
  if (cfg.lookupValue(config_string, dval) == false) {
    if (required) {
      std::cerr << "Could not find value: " << config_string
                << " in default config" << std::endl;
      throw LookupException();
      return EXIT_FAILURE;
    }
  } else {
    *val = dval;
  }
  return EXIT_SUCCESS;
}

int ConfigLookupBool(const char *config_string, const libconfig::Config &cfg,
                     const bool required, bool *val) {
  bool bval;
  if (cfg.lookupValue(config_string, bval) == false) {
    if (required) {
      std::cerr << "Could not find value: " << config_string
                << " in default config" << std::endl;
      throw LookupException();
      return EXIT_FAILURE;
    }
  } else {
    *val = bval;
  }
  return EXIT_SUCCESS;
}

int ConfigLookupCString(const char *config_string, const libconfig::Config &cfg,
                        const bool required, char **val) {
  std::string sval;

  if (cfg.lookupValue(config_string, sval) == false) {
    if (required) {
      std::cerr << "Could not find value: " << config_string
                << " in default config" << std::endl;
      throw LookupException();
      return EXIT_FAILURE;
    }
  } else {
    *val = new char[sval.size() + 1];
    std::strcpy(*val, sval.c_str());
  }
  return EXIT_SUCCESS;
}

int ConfigLookupString(const char *config_string, const libconfig::Config &cfg,
                       const bool required, std::string &sval) {
  // std::cout << "Looking up: " << config_string
  //            << " in default config" << std::endl;
  if (cfg.lookupValue(config_string, sval) == false) {
    if (required) {
      std::cerr << "Could not find value: " << config_string
                << " in default config" << std::endl;
      throw LookupException();
      return EXIT_FAILURE;
    }
  }
  return EXIT_SUCCESS;
}

int AssignConfigParams(const libconfig::Config &cfg, const bool required,
                       Param *params) {
  try {
    ConfigLookupInt("SAMPLING_FREQUENCY", cfg, required, &(params->fs));

    params->frame_length_long = (int)round(
        50 / 1000.0 * (double)params->fs);  // Hard coded value of 50ms.

    double shift_ms = -1.0;
    ConfigLookupDouble("FRAME_SHIFT", cfg, required, &(shift_ms));
    if (required || shift_ms > 0)
      params->frame_shift = (int)round(shift_ms / 1000.0 * (double)params->fs);

    double frame_ms = -1.0;
    ConfigLookupDouble("FRAME_LENGTH", cfg, required, &(frame_ms));
    if (required || frame_ms > 0)
      params->frame_length = (int)round(frame_ms / 1000.0 * (double)params->fs);

    frame_ms = -1.0;
    ConfigLookupDouble("UNVOICED_FRAME_LENGTH", cfg, required, &(frame_ms));

    if (required || frame_ms > 0) {
      params->frame_length_unvoiced =
          (int)round(frame_ms / 1000.0 * (double)params->fs);
    }

    ConfigLookupBool("USE_EXTERNAL_F0", cfg, required,
                     &(params->use_external_f0));

    ConfigLookupString("EXTERNAL_F0_FILENAME", cfg, required,
                       params->external_f0_filename);

    ConfigLookupBool("USE_EXTERNAL_GCI", cfg, required,
                     &(params->use_external_gci));

    ConfigLookupString("EXTERNAL_GCI_FILENAME", cfg, required,
                       params->external_gci_filename);

    ConfigLookupBool("USE_EXTERNAL_LSF_VT", cfg, required,
                     &(params->use_external_lsf_vt));

    ConfigLookupString("EXTERNAL_LSF_VT_FILENAME", cfg, required,
                       params->external_lsf_vt_filename);

    ConfigLookupDouble("GIF_PRE_EMPHASIS_COEFFICIENT", cfg, required,
                       &(params->gif_pre_emphasis_coefficient));

    ConfigLookupDouble("UNVOICED_PRE_EMPHASIS_COEFFICIENT", cfg, false,
                       &(params->unvoiced_pre_emphasis_coefficient));

    ConfigLookupBool("USE_ITERATIVE_GIF", cfg, required,
                     &(params->use_iterative_gif));

    ConfigLookupInt("LPC_ORDER_GLOT_IAIF", cfg, required,
                    &(params->lpc_order_glot_iaif));

    ConfigLookupInt("LPC_ORDER_VT", cfg, required, &(params->lpc_order_vt));

    ConfigLookupInt("LPC_ORDER_GLOT", cfg, required, &(params->lpc_order_glot));

    ConfigLookupDouble("WARPING_LAMBDA_VT", cfg, required,
                       &(params->warping_lambda_vt));

    ConfigLookupDouble("AME_DURATION_QUOTIENT", cfg, required,
                       &(params->ame_duration_quotient));

    ConfigLookupDouble("AME_POSITION_QUOTIENT", cfg, required,
                       &(params->ame_position_quotient));

    ConfigLookupBool("QMF_SUBBAND_ANALYSIS", cfg, required,
                     &(params->qmf_subband_analysis));

    ConfigLookupDouble("MAX_PULSE_LEN_DIFF", cfg, required,
                       &(params->max_pulse_len_diff));

    ConfigLookupInt("PAF_PULSE_LENGTH", cfg, required,
                    &(params->paf_pulse_length));

    ConfigLookupBool("USE_PULSE_INTERPOLATION", cfg, required,
                     &(params->use_pulse_interpolation));

    ConfigLookupBool("USE_WSOLA", cfg, false, &(params->use_wsola));

    ConfigLookupBool("USE_WSOLA_PITCH_SHIFT", cfg, false,
                     &(params->use_wsola_pitch_shift));

    ConfigLookupBool("USE_PAF_UNVOICED", cfg, false,
                     &(params->use_paf_unvoiced_synthesis));

    ConfigLookupBool("USE_VELVET_UNVOICED_PAF", cfg, false,
                     &(params->use_velvet_unvoiced_paf));

    ConfigLookupBool("HP_FILTERING", cfg, required,
                     &(params->use_highpass_filtering));

    ConfigLookupBool("USE_WAVEFORMS_DIRECTLY", cfg, required,
                     &(params->use_waveforms_directly));

    ConfigLookupBool("EXTRACT_F0", cfg, required, &(params->extract_f0));

    ConfigLookupBool("EXTRACT_GAIN", cfg, required, &(params->extract_gain));
    
    ConfigLookupBool("EXTRACT_LSF_VT", cfg, required,
                     &(params->extract_lsf_vt));

    ConfigLookupBool("EXTRACT_LSF_GLOT", cfg, required,
                     &(params->extract_lsf_glot));

    ConfigLookupBool("EXTRACT_HNR", cfg, required, &(params->extract_hnr));

    ConfigLookupBool("EXTRACT_INFOFILE", cfg, required,
                     &(params->extract_infofile));

    ConfigLookupBool("EXTRACT_GLOTTAL_EXCITATION", cfg, required,
                     &(params->extract_glottal_excitation));

    ConfigLookupBool("EXTRACT_ORIGINAL_SIGNAL", cfg, false,
                     &(params->extract_original_signal));

    ConfigLookupBool("EXTRACT_GCI_SIGNAL", cfg, required,
                     &(params->extract_gci_signal));

    ConfigLookupBool("EXTRACT_PULSES_AS_FEATURES", cfg, required,
                     &(params->extract_pulses_as_features));

    ConfigLookupInt("LPC_ORDER_QMF1", cfg, required,
                    &(params->lpc_order_vt_qmf1));

    ConfigLookupInt("LPC_ORDER_QMF2", cfg, required,
                    &(params->lpc_order_vt_qmf2));

    ConfigLookupDouble("F0_MIN", cfg, required, &(params->f0_min));

    ConfigLookupDouble("F0_MAX", cfg, required, &(params->f0_max));

    ConfigLookupDouble("VOICING_THRESHOLD", cfg, required,
                       &(params->voicing_threshold));

    ConfigLookupDouble("ZCR_THRESHOLD", cfg, required,
                       &(params->zcr_threshold));

    ConfigLookupDouble("RELATIVE_F0_THRESHOLD", cfg, required,
                       &(params->relative_f0_threshold));

    ConfigLookupInt("HNR_ORDER", cfg, required, &(params->hnr_order));

    ConfigLookupDouble("SPEED_SCALE", cfg, required, &(params->speed_scale));

    ConfigLookupDouble("PITCH_SCALE", cfg, required, &(params->pitch_scale));

    ConfigLookupDouble("POSTFILTER_COEFFICIENT", cfg, required,
                       &(params->postfilter_coefficient));

    ConfigLookupDouble("POSTFILTER_COEFFICIENT_GLOT", cfg, false,
                       &(params->postfilter_coefficient_glot));

    ConfigLookupBool("USE_POSTFILTERING", cfg, required,
                     &(params->use_postfiltering));

    ConfigLookupBool("USE_TRAJECTORY_SMOOTHING", cfg, required,
                     &(params->use_trajectory_smoothing));

    ConfigLookupBool("USE_GENERIC_ENVELOPE", cfg, required,
                     &(params->use_generic_envelope));

    ConfigLookupBool("USE_SPECTRAL_MATCHING", cfg, required,
                     &(params->use_spectral_matching));

    ConfigLookupInt("LSF_VT_SMOOTH_LEN", cfg, required,
                    &(params->lsf_vt_smooth_len));

    ConfigLookupInt("LSF_GLOT_SMOOTH_LEN", cfg, required,
                    &(params->lsf_glot_smooth_len));

    ConfigLookupInt("GAIN_SMOOTH_LEN", cfg, required,
                    &(params->gain_smooth_len));

    ConfigLookupInt("HNR_SMOOTH_LEN", cfg, required, &(params->hnr_smooth_len));

    ConfigLookupDouble("NOISE_GAIN_VOICED", cfg, required,
                       &(params->noise_gain_voiced));

    ConfigLookupDouble("NOISE_LOW_FREQ_LIMIT_VOICED", cfg, required,
                       &(params->noise_low_freq_limit_voiced));

    ConfigLookupDouble("NOISE_GAIN_UNVOICED", cfg, required,
                       &(params->noise_gain_unvoiced));

    ConfigLookupString("DNN_WEIGHT_PATH", cfg, required,
                       params->dnn_path_basename);

    ConfigLookupBool("USE_PAF_ENERGY_NORM", cfg, required,
                     &(params->use_paf_energy_normalization));

    ConfigLookupBool("NOISE_GATED_SYNTHESIS", cfg, false,
                     &(params->noise_gated_synthesis));

    ConfigLookupDouble("NOISE_GATE_LIMIT_DB", cfg, false,
                       &(params->noise_gate_limit_db));

    ConfigLookupDouble("NOISE_REDUCTION_DB", cfg, false,
                       &(params->noise_reduction_db));

    double update_interval_ms = 0;
    ConfigLookupDouble("FILTER_UPDATE_INTERVAL_VT", cfg, required,
                       &(update_interval_ms));
    if (required || update_interval_ms > 0) {
      params->filter_update_interval_vt =
          (int)round(update_interval_ms / 1000.0 * (double)params->fs);
    }

    update_interval_ms = 0;
    ConfigLookupDouble("FILTER_UPDATE_INTERVAL_SPECMATCH", cfg, required,
                       &(update_interval_ms));
    if (required || update_interval_ms > 0) {
      params->filter_update_interval_specmatch =
          (int)round(update_interval_ms / 1000.0 * (double)params->fs);
    }

    ConfigLookupString("DATA_DIRECTORY", cfg, required, params->data_directory);

    ConfigLookupBool("SAVE_TO_DATADIR_ROOT", cfg, required,
                     &(params->save_to_datadir_root));

    ConfigLookupBool("USE_PITCH_SYNCHRONOUS_ANALYSIS", cfg, required,
                     &(params->use_pitch_synchronous_analysis));

    /* Lookup for parameter directory paths, always optional */
    ConfigLookupString("DIR_GAIN", cfg, false, params->dir_gain);
    ConfigLookupString("DIR_F0", cfg, false, params->dir_f0);
    ConfigLookupString("DIR_LSF_VT", cfg, false, params->dir_lsf);
    ConfigLookupString("DIR_LSF_GLOT", cfg, false, params->dir_lsfg);
    ConfigLookupString("DIR_HNR", cfg, false, params->dir_hnr);
    ConfigLookupString("DIR_PULSES_AS_FEATURES", cfg, false, params->dir_paf);
    ConfigLookupString("DIR_EXCITATION", cfg, false, params->dir_exc);
    ConfigLookupString("DIR_SPECTRUM", cfg, false, params->dir_sp);
    ConfigLookupString("DIR_SYN", cfg, false, params->dir_syn);

    /* Lookup for parameter extensions,  optional */
    ConfigLookupString("EXT_GAIN", cfg, false, params->extension_gain);
    ConfigLookupString("EXT_F0", cfg, false, params->extension_f0);
    ConfigLookupString("EXT_LSF_VT", cfg, false, params->extension_lsf);
    ConfigLookupString("EXT_LSF_GLOT", cfg, false, params->extension_lsfg);
    ConfigLookupString("EXT_HNR", cfg, false, params->extension_hnr);
    ConfigLookupString("EXT_PULSES_AS_FEATURES", cfg, false,
                       params->extension_paf);
    ConfigLookupString("EXT_EXCITATION", cfg, false, params->extension_exc);
    ConfigLookupString("EXT_EXCITATION_ORIG", cfg, false,
                       params->extension_src);

    /* Read enum style configurations */
    std::string str;

    /* Data Format */
    str.clear();
    ConfigLookupString("DATA_TYPE", cfg, required, str);
    if (required || str != "") {
      if (str == "ASCII")
        params->data_type = ASCII;
      else if (str == "DOUBLE")
        params->data_type = DOUBLE;
      else if (str == "FLOAT")
        params->data_type = FLOAT;
      else {
        std::cerr << "Error: DATA_TYPE must be either ASCII, DOUBLE of FLOAT"
                  << std::endl;
        return EXIT_FAILURE;
      }
    }

    /* Signal Polarity */
    str.clear();
    ConfigLookupString("SIGNAL_POLARITY", cfg, required, str);
    if (required || str != "") {
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
    ConfigLookupString("LP_WEIGHTING_FUNCTION", cfg, required, str);
    if (required || str != "") {
      if (str == "NONE")
        params->lp_weighting_function = NONE;
      else if (str == "AME")
        params->lp_weighting_function = AME;
      else if (str == "STE")
        params->lp_weighting_function = STE;
      else
        return EXIT_FAILURE;
    }

    /* Excitation method */
    str.clear();
    ConfigLookupString("EXCITATION_METHOD", cfg, required, str);
    if (required || str != "") {
      if (str == "SINGLE_PULSE")
        params->excitation_method = SINGLE_PULSE_EXCITATION;
      else if (str == "DNN_GENERATED")
        params->excitation_method = DNN_GENERATED_EXCITATION;
      else if (str == "PULSES_AS_FEATURES")
        params->excitation_method = PULSES_AS_FEATURES_EXCITATION;
      else if (str == "EXTERNAL")
        params->excitation_method = EXTERNAL_EXCITATION;
      else if (str == "IMPULSE")
        params->excitation_method = IMPULSE_EXCITATION;
      else {
        std::cerr << "Error: invalid excitation method flag \"" << str << "\""
                  << std::endl;
        std::cerr << "Valid options are SINGLE_PULSE / DNN_GENERATED / "
                     "PULSES_AS_FEATURES / EXTERNAL/ IMPULSE"
                  << std::endl;
        return EXIT_FAILURE;
      }
    }

    /* require external excitation filename if used */
    // TODO: should not be required in config_default
    if (params->excitation_method == EXTERNAL_EXCITATION) {
      ConfigLookupString("EXTERNAL_EXCITATION_FILENAME", cfg, required,
                         params->external_excitation_filename);
    }

    /* PSOLA window for synthesis */
    str.clear();
    ConfigLookupString("PSOLA_WINDOW", cfg, required, str);
    if (required || str != "") {
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
    ConfigLookupString("PAF_WINDOW", cfg, required, str);
    if (required || str != "") {
      if (str == "NONE")
        params->paf_analysis_window = RECT;
      else if (str == "COSINE")
        params->paf_analysis_window = COSINE;
      else if (str == "HANN")
        params->paf_analysis_window = HANN;
      else
        return EXIT_FAILURE;
    }

    std::cout << "read config succesfully" << std::endl;
    return EXIT_SUCCESS;

  } catch (LookupException &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  std::cout << "read config succesfully" << std::endl;
  return EXIT_SUCCESS;
}

int ReadConfig(const char *filename, const bool required, Param *params) {
  libconfig::Config cfg;
  /* Read the file. If there is an error, report it and exit. */
  try {
    cfg.readFile(filename);
  } catch (const libconfig::FileIOException &fioex) {
    std::cerr << "I/O error while reading file: " << filename << std::endl;
    return (EXIT_FAILURE);
  } catch (const libconfig::ParseException &pex) {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
    return (EXIT_FAILURE);
  }

  return AssignConfigParams(cfg, required, params);
  // return EXIT_SUCCESS;
}
