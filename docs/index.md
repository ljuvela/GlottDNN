# GlottDNN Vocoder



The GlottDNN package contains two main parts:

1) The glottal vocoder written in C++
   - Dependencies: libsndfile, libgsl, libconfig

2) Python scripts for vocoder analysis, synthesis and training a DNN excitation model:
   - Dependencies: python, numpy, theano


## Installing

The vocoder C++ code has the following library dependencies:
- `libgsl` (GNU scientific library), for basic linear algebra and FFT etc.
- `libsndfile` for reading and writing audio files
- `libconfig++` for reading structured configuration files

Usually the best way to install the dependencies is with the system package manager. For example, in Ubuntu use `apt-get` install the packages `libgsl0-dev`, `libsndfile1-dev`, `libconfig++-dev`

The C++ part uses a standard GNU autotools build system. To compile the vocoder, run the following commands in this directory:
``` shell
   ./configure
   make
```

Since the build targets are rather generically named `Analysis` and `Synthesis`, you might not want them in your default system PATH. To choose 
``` shell
   ./configure --prefix=/your/install/path/bin
   make install
```

## Support

When in trouble, open an issue at GitHub
https://github.com/ljuvela/GlottDNN/issues

For more examples and explanation, check the documentation in https://aalto-speech.github.io/GlottDNN/ 



## Licence

Copyright 2016-2018 Lauri Juvela and Manu Airaksinen

See LICENCE and NOTICE for more information