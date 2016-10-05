/***********************************************/
/*                 INCLUDE                     */
/***********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <vector>

#include <iostream>
#include <iomanip>
#include <cstdlib>

#include	<cstdio>
#include	<cstring>

#include <gslwrap/vector_double.h>

#include "Filters.h"
#include "definitions.h"
#include "FileIo.h"
#include "ReadConfig.h"
#include "AnalysisFunctions.h"

#include "SpFunctions.h"
#include "DebugUtils.h"


/*******************************************************************/
/*                          MAIN                                   */
/*******************************************************************/

int main(int argc, char *argv[]) {


	const char *wav_filename = argv[1];
	const char *default_config_filename = argv[2];
	const char *user_config_filename = argv[3];

	Param params;
	if (ReadConfig(default_config_filename, true, &params) == EXIT_FAILURE)
		return EXIT_FAILURE;
	if (argc > 3) {
		if (ReadConfig(user_config_filename, false, &params) == EXIT_FAILURE)
			return EXIT_FAILURE;
	}

	//double foo[] = {1,2,3};
    std::cout << "Foo" << std::endl;

	//create_file(fname, SF_FORMAT_WAV | SF_FORMAT_PCM_16) ;
	/* Read sound file and allocate data */
	AnalysisData data;
	ReadFile(wav_filename, &(data.signal), &params);
	data.AllocateData(params);

	/* F0 Analysis */
	GetF0(params, data.signal, &(data.fundf));

	GetGci(params, data.signal, &(data.gci_inds));

    std::cout << "Foo" << std::endl;

	GetGain(params, data.signal, &(data.frame_energy));

	SpectralAnalysis(params, data, &(data.poly_vocal_tract));

	// Process VT parameters?

    std::cout << "Foo" << std::endl;



	/* Finish */
	printf("Finished analysis.\n\n");
	return EXIT_SUCCESS;
}

/***********/
/*   EOF   */
/***********/
