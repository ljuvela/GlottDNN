#ifndef WORKFLOW_H_
#define WORKFLOW_H_

#include <string>
#include "definitions.h"

int RunAnalysis(const std::string &wav_filename,
                const std::string &default_config_filename,
                const std::string &user_config_filename = "");
int RunSynthesis(const std::string &filename,
                 const std::string &default_config_filename,
                 const std::string &user_config_filename = "");
int AnalyzeSignal(const std::string &default_config_filename,
                  const std::string &user_config_filename,
                  const gsl::vector &signal, AnalysisData *data);
int SynthesizeData(const std::string &default_config_filename,
                   const std::string &user_config_filename,
                   SynthesisData *data, gsl::vector *signal,
                   gsl::vector *excitation);

#endif
