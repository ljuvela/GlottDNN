#ifndef WORKFLOW_H_
#define WORKFLOW_H_

#include <string>

int RunAnalysis(const std::string &wav_filename,
                const std::string &default_config_filename,
                const std::string &user_config_filename = "");
int RunSynthesis(const std::string &filename,
                 const std::string &default_config_filename,
                 const std::string &user_config_filename = "");

#endif
