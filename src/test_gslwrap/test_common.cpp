/*
 * Copyright (c) 2001, ULP-IPB Strasbourg.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 */
#include <vector>
#include <string>

extern void Histogram();
extern void Vector();
extern void VectorFloat();
extern void VectorDiagonalView();
extern void VectorView();
extern void VectorView2();
extern void VectorView3();
extern void GSLFunctionCall();
extern void RandomNumberGenerator();
extern void LUInvertAndDecomp();
extern void Histogram();
extern void Histogram();
extern void Histogram();
extern void OneDimMinimiserTest();
extern void MultDimMinimiserTest();

//extern void PCAppearanceModel_Test();
#ifndef __HP_aCC
using std::string;
using std::vector;
#endif


int
main(int argc,char **argv)
{
	std::vector<string> args;
	for(int i=1;i<argc;i++){args.push_back(argv[i]);}
	string testName=args[0].substr(string("testc_").size());

//	fprintf(stderr,"testing ... \"%s\" :",testName.c_str());

	if (testName=="Histogram"               ) Histogram();
	if (testName=="Vector"               ) Vector();
	if (testName=="VectorFloat"               ) VectorFloat();
	if (testName=="VectorView"               ) VectorView();
	if (testName=="VectorDiagonalView"               ) VectorDiagonalView();
	if (testName=="VectorView2"               ) VectorView2();
	if (testName=="VectorView3"               ) VectorView3();
	if (testName=="GSLFunctionCall"               ) GSLFunctionCall();
	if (testName=="RandomNumberGenerator"               ) RandomNumberGenerator();
	if (testName=="LUInvertAndDecomp"               ) LUInvertAndDecomp();
	if (testName=="OneDimMinimiserTest"               ) OneDimMinimiserTest();
	if (testName=="MultDimMinimiserTest"               ) MultDimMinimiserTest();

  	return 0;

}
