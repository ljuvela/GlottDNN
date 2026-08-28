#include <pybind11/pybind11.h>

#include "Workflow.h"

namespace py = pybind11;

PYBIND11_MODULE(glottdnn_cpp, module) {
   module.doc() = "Python bindings for the GlottDNN C++ vocoder";

   py::module analysis = module.def_submodule(
       "analysis", "File-based acoustic analysis operations");
   analysis.def("run", &RunAnalysis,
                py::arg("wav_filename"),
                py::arg("default_config_filename"),
                py::arg("user_config_filename") = "");

   py::module synthesis = module.def_submodule(
       "synthesis", "File-based waveform synthesis operations");
   synthesis.def("run", &RunSynthesis,
                 py::arg("filename"),
                 py::arg("default_config_filename"),
                 py::arg("user_config_filename") = "");

   module.def_submodule(
       "signal_processing",
       "Reserved namespace for future individual signal-processing bindings");
}
