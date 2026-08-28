#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <stdexcept>

#include "Workflow.h"
#include "SignalProcessingBindings.h"

namespace py = pybind11;

namespace {
gsl::vector ToVector(const py::handle &value) {
   auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
   if (!array) throw std::invalid_argument("expected numeric vector data");
   auto view = array.request();
   if (view.ndim != 1) throw std::invalid_argument("expected one-dimensional data");
   gsl::vector result(static_cast<size_t>(view.shape[0]));
   const double *values = static_cast<const double *>(view.ptr);
   for (size_t i = 0; i < result.size(); ++i) result(i) = values[i];
   return result;
}
gsl::matrix ToMatrix(const py::handle &value) {
   auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
   if (!array) throw std::invalid_argument("expected numeric matrix data");
   auto view = array.request();
   if (view.ndim != 2) throw std::invalid_argument("expected two-dimensional data");
   gsl::matrix result(static_cast<size_t>(view.shape[0]),
                      static_cast<size_t>(view.shape[1]));
   const double *values = static_cast<const double *>(view.ptr);
   for (size_t i = 0; i < result.get_rows(); ++i)
      for (size_t j = 0; j < result.get_cols(); ++j)
         result(i, j) = values[i * result.get_cols() + j];
   return result;
}
gsl::vector_int ToIntVector(const py::handle &value) {
   auto array = py::array_t<int, py::array::c_style | py::array::forcecast>::ensure(value);
   if (!array) throw std::invalid_argument("expected integer vector data");
   auto view = array.request();
   if (view.ndim != 1) throw std::invalid_argument("expected one-dimensional indices");
   gsl::vector_int result(static_cast<size_t>(view.shape[0]));
   const int *values = static_cast<const int *>(view.ptr);
   for (size_t i = 0; i < result.size(); ++i) result[i] = values[i];
   return result;
}
py::array_t<double> Vector(const gsl::vector &value) {
   py::array_t<double> result(value.size());
   auto view = result.mutable_unchecked<1>();
   for (size_t i = 0; i < value.size(); ++i) view(i) = value(i);
   return result;
}
py::array_t<double> Matrix(const gsl::matrix &value) {
   py::array_t<double> result({value.get_rows(), value.get_cols()});
   auto view = result.mutable_unchecked<2>();
   for (size_t i = 0; i < value.get_rows(); ++i)
      for (size_t j = 0; j < value.get_cols(); ++j) view(i, j) = value(i, j);
   return result;
}
py::array_t<int> IntVector(const gsl::vector_int &value) {
   py::array_t<int> result(value.size());
   auto view = result.mutable_unchecked<1>();
   for (size_t i = 0; i < value.size(); ++i) view(i) = value[i];
   return result;
}
py::dict AnalysisResult(const AnalysisData &data) {
   py::dict result;
   result["signal"] = Vector(data.signal);
   result["fundf"] = Vector(data.fundf);
   result["frame_energy"] = Vector(data.frame_energy);
   result["gci_inds"] = IntVector(data.gci_inds);
   result["source_signal"] = Vector(data.source_signal);
   result["source_signal_iaif"] = Vector(data.source_signal_iaif);
   result["poly_vocal_tract"] = Matrix(data.poly_vocal_tract);
   result["lsf_vocal_tract"] = Matrix(data.lsf_vocal_tract);
   result["poly_glot"] = Matrix(data.poly_glot);
   result["lsf_glot"] = Matrix(data.lsf_glot);
   result["excitation_pulses"] = Matrix(data.excitation_pulses);
   result["hnr_glot"] = Matrix(data.hnr_glot);
   return result;
}
SynthesisData SynthesisInput(const py::dict &values) {
   SynthesisData data;
   data.signal = ToVector(values["signal"]);
   data.fundf = ToVector(values["fundf"]);
   data.frame_energy = ToVector(values["frame_energy"]);
   data.poly_vocal_tract = ToMatrix(values["poly_vocal_tract"]);
   data.lsf_vocal_tract = ToMatrix(values["lsf_vocal_tract"]);
   data.poly_glot = ToMatrix(values["poly_glot"]);
   data.lsf_glot = ToMatrix(values["lsf_glot"]);
   data.excitation_pulses = ToMatrix(values["excitation_pulses"]);
   data.hnr_glot = ToMatrix(values["hnr_glot"]);
   return data;
}
}

PYBIND11_MODULE(glottdnn_cpp, module) {
   module.doc() = "Python bindings for the GlottDNN C++ vocoder";

   py::module analysis = module.def_submodule(
       "analysis", "File-based acoustic analysis operations");
   analysis.def("run", &RunAnalysis,
                py::arg("wav_filename"),
                py::arg("default_config_filename"),
                py::arg("user_config_filename") = "");
    analysis.def("run_array", [](py::array signal, const std::string &config,
                                const std::string &user_config) {
       AnalysisData data;
       if (AnalyzeSignal(config, user_config, ToVector(signal), &data) != 0)
          throw std::runtime_error("analysis failed");
       return AnalysisResult(data);
    }, py::arg("signal"), py::arg("default_config_filename"),
       py::arg("user_config_filename") = "");

   py::module synthesis = module.def_submodule(
       "synthesis", "File-based waveform synthesis operations");
   synthesis.def("run", &RunSynthesis,
                 py::arg("filename"),
                 py::arg("default_config_filename"),
                 py::arg("user_config_filename") = "");
    synthesis.def("run_data", [](const py::dict &values, const std::string &config,
                                const std::string &user_config) {
       SynthesisData data = SynthesisInput(values);
       gsl::vector signal;
       gsl::vector excitation;
       if (SynthesizeData(config, user_config, &data, &signal, &excitation) != 0)
          throw std::runtime_error("synthesis failed");
       py::dict result;
       result["signal"] = Vector(signal);
       result["excitation_signal"] = Vector(excitation);
       return result;
    }, py::arg("data"), py::arg("default_config_filename"),
       py::arg("user_config_filename") = "");

   py::module signal_processing = module.def_submodule(
       "signal_processing",
       "NumPy bindings for signal-processing functions");
   BindSignalProcessing(signal_processing);
}
