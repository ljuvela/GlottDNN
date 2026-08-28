cmake_minimum_required(VERSION 3.15)

project(GlottDNN VERSION 1.2 LANGUAGES CXX)

include(GNUInstallDirs)

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

add_executable(Analysis
    src/glott/Analysis.cpp
    src/glott/AnalysisFunctions.cpp
    src/glott/PitchEstimation.cpp
    src/glott/QmfFunctions.cpp
    ${GLOTT_SOURCES}
)

add_executable(Synthesis
    src/glott/Synthesis.cpp
    src/glott/DnnClass.cpp
    src/glott/SynthesisFunctions.cpp
    ${GLOTT_SOURCES}
)

add_executable(LsfPostFilter
    src/glott/LsfPostFilterMain.cpp
    src/glott/DnnClass.cpp
    src/glott/SynthesisFunctions.cpp
    ${GLOTT_SOURCES}
)

foreach(target IN ITEMS Analysis Synthesis LsfPostFilter)
    target_link_libraries(${target}
        PRIVATE
            gslwrap
            ${GSL_LIBRARY}
            ${GSL_CBLAS_LIBRARY}
            ${LIBCONFIG_LIBRARY}
            ${SNDFILE_LIBRARY}
    )
    target_include_directories(${target}
        PRIVATE
            ${LIBCONFIG_INCLUDE_DIR}
            ${SNDFILE_INCLUDE_DIR}
    )
    install(TARGETS ${target}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endforeach()
