cmake_minimum_required(VERSION 3.15)

project(GlottDNN VERSION 1.2 LANGUAGES CXX)

include(GNUInstallDirs)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra -pedantic)

find_path(GSL_INCLUDE_DIR gsl/gsl_math.h)
find_library(GSL_LIBRARY NAMES gsl)
find_library(GSL_CBLAS_LIBRARY NAMES gslcblas)
find_path(LIBCONFIG_INCLUDE_DIR libconfig.h++)
find_library(LIBCONFIG_LIBRARY NAMES config++)
find_path(SNDFILE_INCLUDE_DIR sndfile.hh)
find_library(SNDFILE_LIBRARY NAMES sndfile)

foreach(dependency IN ITEMS
        GSL_INCLUDE_DIR GSL_LIBRARY GSL_CBLAS_LIBRARY
        LIBCONFIG_INCLUDE_DIR LIBCONFIG_LIBRARY
        SNDFILE_INCLUDE_DIR SNDFILE_LIBRARY)
    if(NOT ${dependency})
        message(FATAL_ERROR "Could not find required dependency: ${dependency}")
    endif()
endforeach()

add_library(gslwrap STATIC
    src/gslwrap/matrix_double.cc
    src/gslwrap/matrix_float.cc
    src/gslwrap/matrix_int.cc
    src/gslwrap/random_generator.cc
    src/gslwrap/vector_double.cc
    src/gslwrap/vector_float.cc
    src/gslwrap/vector_int.cc
)

target_include_directories(gslwrap
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${GSL_INCLUDE_DIR}
)

set(GLOTT_SOURCES
    src/glott/definitions.cpp
    src/glott/ReadConfig.cpp
    src/glott/FileIo.cpp
    src/glott/InverseFiltering.cpp
    src/glott/ComplexVector.cpp
    src/glott/SpFunctions.cpp
    src/glott/Utils.cpp
)

add_library(glott_core STATIC
    src/glott/AnalysisFunctions.cpp
    src/glott/PitchEstimation.cpp
    src/glott/QmfFunctions.cpp
    src/glott/DnnClass.cpp
    src/glott/SynthesisFunctions.cpp
    src/glott/Workflow.cpp
    ${GLOTT_SOURCES}
)

target_include_directories(glott_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GSL_INCLUDE_DIR}
    ${LIBCONFIG_INCLUDE_DIR}
    ${SNDFILE_INCLUDE_DIR}
)
target_link_libraries(glott_core PUBLIC
    gslwrap
    ${GSL_LIBRARY}
    ${GSL_CBLAS_LIBRARY}
    ${LIBCONFIG_LIBRARY}
    ${SNDFILE_LIBRARY}
)

add_executable(Analysis src/glott/Analysis.cpp)
add_executable(Synthesis src/glott/Synthesis.cpp)
add_executable(LsfPostFilter src/glott/LsfPostFilterMain.cpp)
target_link_libraries(Analysis PRIVATE glott_core)
target_link_libraries(Synthesis PRIVATE glott_core)
target_link_libraries(LsfPostFilter PRIVATE glott_core)

foreach(target IN ITEMS Analysis Synthesis LsfPostFilter)
    install(TARGETS ${target}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endforeach()

pybind11_add_module(glottdnn_cpp src/glott/bindings.cpp)
target_link_libraries(glottdnn_cpp PRIVATE
    glott_core
)
target_include_directories(glottdnn_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GSL_INCLUDE_DIR}
    ${LIBCONFIG_INCLUDE_DIR}
    ${SNDFILE_INCLUDE_DIR}
)
install(TARGETS glottdnn_cpp
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
