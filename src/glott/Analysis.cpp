/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015 Tuomo Raitio
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *
 *
 *
 * <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
 *              GlottHMM Speech Parameter Extractor
 * <><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>
 *
 * This program reads a speech file and extracts speech
 * parameters using glottal inverse filtering.
 *
 * This program has been written in Aalto University,
 * Department of Signal Processign and Acoustics, Espoo, Finland
 *
 * Authors: Tuomo Raitio (2008-2015), Lauri Juvela (2015-)
 * Acknowledgements: Antti Suni, Paavo Alku, Martti Vainio
 *
 * File Analysis.c
 * Version: 2.0
 *
 */



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


using namespace std;




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
	//std::cout << sizeof(foo) / sizeof(double) << std::endl;

	//create_file(fname, SF_FORMAT_WAV | SF_FORMAT_PCM_16) ;
	/* Read sound file and allocate data */
	AnalysisData data;
	ReadFile(wav_filename, &(data.signal), &params);
	data.AllocateData(params);

	/* F0 Analysis */
	GetF0(params, data.signal, &(data.fundf));

	GetGci(params, data.signal, &(data.gci_inds));


	//VPrint1(data.gci_signal);
	GetGain(params, data.signal, &(data.frame_energy));

	SpectralAnalysis(params, data, &(data.poly_vocal_tract));

	// Process VT parameters?




	/* Finish */
	printf("Finished analysis.\n\n");
	return EXIT_SUCCESS;
}

/***********/
/*   EOF   */
/***********/

