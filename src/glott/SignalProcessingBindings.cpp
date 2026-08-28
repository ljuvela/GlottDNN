#include "SignalProcessingBindings.h"

#include <stdexcept>

#include <pybind11/numpy.h>

#include "definitions.h"
#include "SpFunctions.h"

namespace py = pybind11;

namespace {

gsl::vector ToVector(const py::array_t<double, py::array::c_style | py::array::forcecast> &array) {
   auto view = array.request();
   if (view.ndim != 1)
      throw std::invalid_argument("expected a one-dimensional array");
   gsl::vector result(static_cast<size_t>(view.shape[0]));
   const auto *values = static_cast<const double *>(view.ptr);
   for (size_t i = 0; i < result.size(); ++i)
      result(i) = values[i];
   return result;
}

py::array_t<double> FromVector(const gsl::vector &vector) {
   py::array_t<double> result(vector.size());
   auto view = result.mutable_unchecked<1>();
   for (size_t i = 0; i < vector.size(); ++i)
      view(i) = vector(i);
   return result;
}

py::array_t<double> FromMatrix(const gsl::matrix &matrix) {
   py::array_t<double> result({matrix.get_rows(), matrix.get_cols()});
   auto view = result.mutable_unchecked<2>();
   for (size_t i = 0; i < matrix.get_rows(); ++i)
      for (size_t j = 0; j < matrix.get_cols(); ++j)
         view(i, j) = matrix(i, j);
   return result;
}

gsl::matrix ToMatrix(const py::array_t<double, py::array::c_style | py::array::forcecast> &array) {
   auto view = array.request();
   if (view.ndim != 2)
      throw std::invalid_argument("expected a two-dimensional array");
   gsl::matrix result(view.shape[0], view.shape[1]);
   const auto *values = static_cast<const double *>(view.ptr);
   for (size_t i = 0; i < result.get_rows(); ++i)
      for (size_t j = 0; j < result.get_cols(); ++j)
         result(i, j) = values[i * result.get_cols() + j];
   return result;
}

WindowingFunctionType Window(const std::string &name) {
   if (name == "hann") return HANN;
   if (name == "hamming") return HAMMING;
   if (name == "blackman") return BLACKMAN;
   if (name == "cosine") return COSINE;
   if (name == "rect") return RECT;
   throw std::invalid_argument("unsupported window: " + name);
}

}  // namespace

void BindSignalProcessing(py::module_ &module) {
   module.def("_interpolate_linear", [](py::array_t<double> values, size_t size) {
      gsl::vector output;
      InterpolateLinear(ToVector(values), size, &output);
      return FromVector(output);
   }, py::arg("values"), py::arg("size"));
   module.def("_filter", [](py::array_t<double> b, py::array_t<double> a, py::array_t<double> x) {
      gsl::vector output;
      Filter(ToVector(b), ToVector(a), ToVector(x), &output);
      return FromVector(output);
   }, py::arg("b"), py::arg("a"), py::arg("x"));
   module.def("_conv", [](py::array_t<double> first, py::array_t<double> second) {
      return FromVector(Conv(ToVector(first), ToVector(second)));
   }, py::arg("first"), py::arg("second"));
   module.def("_autocorrelation", [](py::array_t<double> frame, int order) {
      gsl::vector output;
      Autocorrelation(ToVector(frame), order, &output);
      return FromVector(output);
   }, py::arg("frame"), py::arg("order"));
   module.def("_lsf_to_poly", [](py::array_t<double> lsf) {
      gsl::vector output;
      Lsf2Poly(ToVector(lsf), &output);
      return FromVector(output);
   }, py::arg("lsf"));
   module.def("_poly_to_lsf", [](py::array_t<double> poly) {
      gsl::vector output;
      Poly2Lsf(ToVector(poly), &output);
      return FromVector(output);
   }, py::arg("poly"));
   module.def("_lsf_matrix_to_poly", [](py::array_t<double> lsf) {
      gsl::matrix output;
      Lsf2Poly(ToMatrix(lsf), &output);
      return FromMatrix(output);
   }, py::arg("lsf"));
   module.def("_poly_matrix_to_lsf", [](py::array_t<double> poly) {
      gsl::matrix output;
      Poly2Lsf(ToMatrix(poly), &output);
      return FromMatrix(output);
   }, py::arg("poly"));
   module.def("_window", [](const std::string &window, py::array_t<double> frame) {
      gsl::vector output = ToVector(frame);
      ApplyWindowingFunction(Window(window), &output);
      return FromVector(output);
   }, py::arg("window"), py::arg("frame"));
   module.def("_mean", [](py::array_t<double> values) { return getMean(ToVector(values)); });
   module.def("_energy", [](py::array_t<double> values) { return getEnergy(ToVector(values)); });
   module.def("_next_pow2", &NextPow2, py::arg("value"));
   module.def("_linear_to_erb", [](py::array_t<double> values, int fs) {
      gsl::vector output;
      Linear2Erb(ToVector(values), fs, &output);
      return FromVector(output);
   }, py::arg("values"), py::arg("sample_rate"));
   module.def("_erb_to_linear", [](py::array_t<double> values, int fs) {
      gsl::vector output;
      Erb2Linear(ToVector(values), fs, &output);
      return FromVector(output);
   }, py::arg("values"), py::arg("sample_rate"));
   module.def("_median_filter", [](py::array_t<double> values, size_t length) {
      gsl::vector output;
      MedianFilter(ToVector(values), length, &output);
      return FromVector(output);
   }, py::arg("values"), py::arg("length"));
   module.def("_moving_average_filter", [](py::array_t<double> values, size_t length) {
      gsl::vector output;
      MovingAverageFilter(ToVector(values), length, &output);
      return FromVector(output);
   }, py::arg("values"), py::arg("length"));
}
