/*
 * ComplexVector.h
 *
 *  Created on: 7 Oct 2016
 *      Author: ljuvela
 */

#ifndef SRC_GLOTT_COMPLEXVECTOR_H_
#define SRC_GLOTT_COMPLEXVECTOR_H_

#include <cassert>

class ComplexVector {
public:
   ComplexVector();
   ComplexVector(size_t nfft);
   ~ComplexVector();
   bool is_set() {return (real_data!=NULL && imag_data!=NULL && fft_freq_bins>0);};
   double getReal(size_t idx) const {assert(idx<fft_freq_bins); return real_data[idx];};
   gsl::vector getReal() const;
   double getImag(size_t idx) const {assert(idx<fft_freq_bins);return imag_data[idx];};
   gsl::vector getImag() const;
   double getAbs(size_t idx) const;
   gsl::vector getAbs() const;
   double getAng(size_t idx) const;
   gsl::vector getAng() const;
   void setReal(size_t idx, double val) {assert(idx<fft_freq_bins); real_data[idx] = val;};
   void setReal(const gsl::vector &vec);
   void setImag(size_t idx, double val) {assert(idx<fft_freq_bins); imag_data[idx] = val;};
   void setImag(const gsl::vector &vec);
   void setAllReal(double val);
   void setAllImag(double val);
   size_t getSize() const {return fft_freq_bins;};
   void resize(size_t val);
   void operator*=(double x);
   void operator/=(double x);
private:
   size_t fft_freq_bins;
   double *real_data;
   double *imag_data;
   void freeData();

};

#endif /* SRC_GLOTT_COMPLEXVECTOR_H_ */
