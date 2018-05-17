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

#include <gslwrap/vector_double.h>
#include <cmath>
#include "ComplexVector.h"

ComplexVector::ComplexVector() {
   fft_freq_bins = 0;
   real_data = NULL;
   imag_data = NULL;
}

ComplexVector::ComplexVector(size_t freq_bins) {
   fft_freq_bins = freq_bins;
   real_data = new double[freq_bins];
   setAllReal(0.0);
   imag_data = new double[freq_bins];
   setAllImag(0.0);
}

ComplexVector::~ComplexVector() {
   if (real_data)
      delete[] real_data;
   if (imag_data)
      delete[] imag_data;
}

void ComplexVector::setAllReal(double val) {
   if(real_data == NULL) return;
   size_t i;
   for (i=0;i<fft_freq_bins;i++)
      real_data[i] = val;
}

void ComplexVector::setAllImag(double val) {
   if(imag_data == NULL) return;
   size_t i;
   for (i=0;i<fft_freq_bins;i++)
      imag_data[i] = val;
}

gsl::vector ComplexVector::getReal() const {
   gsl::vector vec(this->getSize());
   size_t i;
   for(i=0;i<vec.size();i++)
      vec(i) = real_data[i];
   return vec;
}

gsl::vector ComplexVector::getImag() const {
   gsl::vector vec(this->getSize());
   size_t i;
   for(i=0;i<vec.size();i++)
      vec(i) = imag_data[i];
   return vec;
}

double ComplexVector::getAbs(size_t idx) const {
   assert(idx < getSize());
   return sqrt( getReal(idx)*getReal(idx) + getImag(idx)*getImag(idx));
}

gsl::vector ComplexVector::getAbs() const {
   gsl::vector vec(this->getSize());
   size_t i;
   for(i=0;i<vec.size();i++)
      vec(i) = getAbs(i);
   return vec;
}

double ComplexVector::getAng(size_t idx) const {
   assert(idx < getSize());
   return atan2(getImag(idx), getReal(idx));
}

gsl::vector ComplexVector::getAng() const {
   gsl::vector vec(this->getSize());
   size_t i;
   for(i=0;i<vec.size();i++)
      vec(i) = getAng(i);
   return vec;
}

void ComplexVector::setReal(const gsl::vector &vec) {
   assert(vec.size() == this->getSize());
   size_t i;
   for (i=0;i<vec.size();i++)
      this->setReal(i, vec(i));
}

void ComplexVector::setImag(const gsl::vector &vec) {
   assert(vec.size() == this->getSize());
   size_t i;
   for (i=0;i<vec.size();i++)
      this->setImag(i, vec(i));
}

void ComplexVector::resize(size_t val) {
   if(val == fft_freq_bins) {
      setAllReal(0.0);
      setAllImag(0.0);
      return;
   }
   freeData();
   fft_freq_bins = val;
   real_data = new double[val];
   setAllReal(0.0);
   imag_data = new double[val];
   setAllImag(0.0);
}

void ComplexVector::operator*=(double x) {
   for (size_t i=0; i< this->fft_freq_bins; i++) {
      this->imag_data[i] *= x;
      this->real_data[i] *= x;
   }
}

void ComplexVector::operator/=(double x) {
   for (size_t i=0; i< this->fft_freq_bins; i++) {
      this->imag_data[i] /= x;
      this->real_data[i] /= x;
   }
}

void ComplexVector::freeData() {
   if (real_data)
      delete[] real_data;
   if (imag_data)
      delete[] imag_data;

   fft_freq_bins = 0;
}
