# GlottDNN Vocoder

The GlottDNN package contains two main parts:

1) The glottal vocoder written in C++
   - Dependencies: `libsndfile`, `libgsl`, `libconfig`

2) Python scripts for vocoder analysis, synthesis and training a DNN excitation model:
   - Dependencies: `python3`, `numpy`, `pytorch>=1.1.0`


## Installation

The vocoder C++ code has the following library dependencies:
- `libgsl` (GNU scientific library), for basic linear algebra and FFT etc.
- `libsndfile` for reading and writing audio files
- `libconfig++` for reading structured configuration files

Usually the best way to install the dependencies is with the system package manager. For example, in Ubuntu use `apt-get` install the packages `libgsl0-dev`, `libsndfile1-dev`, `libconfig++-dev`

The C++ part uses a standard GNU autotools build system. To compile the vocoder, run the following commands in the project root directory
``` bash
   ./configure
   make
```

Since the build targets are rather generically named `Analysis` and `Synthesis`, you might not want them in your default system PATH. Use the `--prefix` flag to choose another install path
``` bash
   ./configure --prefix=/your/install/path/bin
   make install
```

Usually `configure` and `make` should be enough, but if the process complains about missing files try running 
``` bash
automake --add-missing
```

### Installation using a conda environment

Conda environments are useful for managing dependencies and keeping a GlottDNN installation contained from the systemwide environment. For more information about managing conda enviroments, see https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html


Create and activate a new conda environment. 
```bash
conda create -n glottdnn
conda activate glottdnn
```

Install dependencies
``` bash
conda install -c conda-forge libsndfile gsl libconfig 
```

Optionally, install pytorch with conda
```bash
conda install pytorch torchaudio -c pytorch
```

Set compiler flags to compiler flags to point to the currently active conda environment. The compiled binaries will be installed in `$CONDA_PREFIX/bin/`  
``` bash
export LDFLAGS=-L$CONDA_PREFIX/lib/ 
export CPPFLAGS=-I$CONDA_PREFIX/include/
./configure --prefix $CONDA_PREFIX
make
make install
```

Library dependencies are linked dynamically, so we need to make sure they are is visible in `LD_LIBRARY_PATH`. Adding an `env_vars.sh` activate script is a convenient way to do this automatically whenever the conda environment is activatied.
```bash
# any shell scripts in these directories are run at activation/deactivation
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
mkdir -p $CONDA_PREFIX/etc/conda/deactivate.d

# modify LD_LIBRARY_PATH at activation
export ACTIVATE_SCRIPT=$CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export OLD_LD_LIBRARY_PATH=${LD_LIBRARY_PATH}' > $ACTIVATE_SCRIPT
echo 'export LD_LIBRARY_PATH='${CONDA_PREFIX}'/lib/:${LD_LIBRARY_PATH}' >> $ACTIVATE_SCRIPT

# restore LD_LIBRARY_PATH state at deactivation
export DEACTIVATE_SCRIPT=$CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=${OLD_LD_LIBRARY_PATH}' > $DEACTIVATE_SCRIPT
echo 'unset OLD_LD_LIBRARY_PATH' >> $DEACTIVATE_SCRIPT
```

## Analysis-synthesis example



These examples assume 16kHz sampling rate audio. Other sampling rates are feasible, but you should change the config accordingly.  


Let's first get a wave file from the Arctic database
``` bash 
#!/bin/bash

URL='http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0001.wav'
DATADIR='./data/tmp'
BASENAME='slt_arctic_a0001'
mkdir -p $DATADIR
wget -O "$DATADIR/$BASENAME.wav" $URL
```

### Acoustic feature analysis

Now run GlottDNN Analysis program with default configuration
``` bash
./src/Analysis "$DATADIR/$BASENAME.wav" ./config/config_default_16k.cfg
```

We should now have the following files 

```
ls ./data/tmp/ 

    ./data/tmp/slt_arctic_a0001.gain
    ./data/tmp/slt_arctic_a0001.lsf
    ./data/tmp/slt_arctic_a0001.slsf
    ./data/tmp/slt_arctic_a0001.hnr
    ./data/tmp/slt_arctic_a0001.pls
    ./data/tmp/slt_arctic_a0001.f0
    ./data/tmp/slt_arctic_a0001.src.wav
```

### Synthesis with single pulse excitation 

First let's run copy synthesis with `SINGLE_PULSE` excitation. This method uses a single fixed glottal pulse, which is modified according to F0 and HNR (similarly to the original GlottHMM vocoder).

``` bash
# Run synthesis with default config
./src/Synthesis "$DATADIR/$BASENAME" ./config/config_default_16k.cfg

# Move generated file
mv "$DATADIR/$BASENAME.syn.wav" "$DATADIR/$BASENAME.syn.sp.wav"    
```

A copy-synthesis wave file should now be at `./data/tmp/slt_arctic_a0001.syn.sp.wav`.
The single pulse excitation will sound somewhat buzzy, so let's try if we can do better.

### Synthesis with original pulses

 We already extracted glottal pulses from the signal and stored them in `./data/tmp/slt_arctic_a0001.pls`. 
 Better quality can be achieved by re-assembling the original pulses using pitch synchronous overlap-add. 

To override some of the default config values, we can create a "user config" file and run Synthesis with two config files

``` bash
# Create user config
CONF_USR="$DATADIR/config_usr.cfg"
echo '# Comment: User config for GlottDNN' > $CONF_USR  
echo 'EXCITATION_METHOD = "PULSES_AS_FEATURES";' >> $CONF_USR
echo 'USE_WSOLA = true;' >> $CONF_USR
echo 'USE_SPECTRAL_MATCHING = false;' >> $CONF_USR
echo 'NOISE_GAIN_VOICED = 0.0;' >> $CONF_USR

# Run synthesis with two config files
./src/Synthesis "$DATADIR/$BASENAME" ./config/config_default_16k.cfg $CONF_USR

# Move generated file
mv "$DATADIR/$BASENAME.syn.wav" "$DATADIR/$BASENAME.syn.paf.wav"       
```

Of course the original pulses are not available in many applications (such as text-to-speech). For this, we can use a trainable excitation model (neural net), which generates the pulses from acoustic features.

## Built-in neural net excitation model 

The present version requires `pytorch>=1.1.0` and all `theano` dependencies have been removed.

Note that the following is a toy example, since we now use only 10 audio files. This example is intended as a quick sanity check and can be easily run on a CPU. For more data and more complex models, a GPU is recommended.


Let's first download some data
```
sh ./dnn_demo/get_data.sh
```

Before we run anything, have a look into
```
./dnn_demo/config_dnn_demo.py
```

Then run the example script by saying
``` bash
python3 ./python/GlottDnnScript.py ./dnn_demo/config_dnn_demo.py
```

The demo script runs vocoder analysis, trains a DNN excitation model, and finally applies copy-synthesis to the samples.
After running, the copy-synthesis results are stored in `./dnn_demo/data/syn` and the original wave files are in `./dnn_demo/data/wav`.

### Python config contents
Prepare a directory structure under and make file lists based on contents of the `wav` sub-directory
``` python
make_dirs = 1
make_scp = 1
```

Optionally, use REAPER for pitch (F0) and GCI analysis. 
Also optionally, use RAPT from SPTK for pitch analysis. These programs need to be installed separately, so this example does not use them. 

``` python
do_reaper_pitch_analysis = 0
do_sptk_pitch_analysis = 0
```

Use GlottDNN to extract glottal vocoder features and pulses for  excitation model training.
``` python
do_glott_vocoder_analysis = 1
```

Package data and train an excitation model for GlottDNN, as supported by the internal implementation. Uses `theano` for training and only supports simple fully connected nets with least squares training.
``` python
make_dnn_train_data = 1
make_dnn_infofile = 1
do_dnn_training = 1
```

Do copy synthesis (using the internal implementation of DNN excitation)
``` python
do_glott_vocoder_synthesis = 1
```

### Improvements from toy example

1) Use more data
2) Experiment with different pitch estimators
    - https://github.com/google/REAPER
    - http://sp-tk.sourceforge.net
3) Use more advanced excitation models
    - https://github.com/ljuvela/multiscale-GAN
    - https://github.com/ljuvela/ResGAN
    - Build your own

## Support

When in trouble, open an issue at GitHub. Others will likely have similar issues and it's best to solve them collectively

https://github.com/ljuvela/GlottDNN/issues

For more examples and explanation, check the documentation in

https://aalto-speech.github.io/GlottDNN/ 

## Licence

Copyright 2016-2018 Lauri Juvela and Manu Airaksinen

This product includes software developed at Aalto University (http://www.aalto.fi/).

Licensed under the Apache License, Version 2.0
See LICENCE and NOTICE for full licence details. 

If you publish work based on GlottDNN, please cite
```
    M. Airaksinen, L. Juvela, B. Bollepalli, J. Yamagishi and P. Alku,
    "A comparison between STRAIGHT, glottal, and sinusoidal vocoding in statistical parametric speech synthesis,"
    in IEEE/ACM Transactions on Audio, Speech, and Language Processing.
    doi: 10.1109/TASLP.2018.2835720. 
```    

The paper also contains a technical details of the vocoder 

If the software is to be deployed in commercial products, permission must be asked from Aalto University 
    (please contact: lauri.juvela@aalto.fi , manu.airaksinen@aalto.fi or paavo.alku@aalto.fi). 

This software distribution also includes third-party C++ wrappers for the GSL library, which are licenced separately under the GPL 3 licence. 
For details, see
    `src/gslwrap/LICENCE`
