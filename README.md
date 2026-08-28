# GlottDNN Vocoder

The GlottDNN package contains a C++ glottal vocoder and Python tools for
vocoder analysis, synthesis, and DNN excitation-model training.

## Installation and build

Conda environments are useful for managing dependencies and keeping a GlottDNN
installation contained from the systemwide environment. The repository includes
an environment specification with the C++ libraries, compiler toolchain, and
Python packages required by the project.

Create and activate the Conda environment, then install the Python package:
```bash
conda env create -f environment.yml
conda activate glottdnn
python -m pip install -e .
```

The editable install automatically configures and builds CMake, installs the
native Python extension, and installs the `Analysis`, `Synthesis`, and
`LsfPostFilter` executables.

Run the tests from the repository root:
```bash
pytest
```
The test suite downloads the small Arctic sample once.

## Analysis-synthesis example



These examples assume 16kHz sampling rate audio. Other sampling rates are feasible, but you should change the config accordingly.  


Let's first get a wave file from the Arctic database
``` bash 
URL='http://festvox.org/cmu_arctic/cmu_arctic/cmu_us_slt_arctic/wav/arctic_a0001.wav'
DATADIR='./data/tmp'
BASENAME='slt_arctic_a0001'
mkdir -p $DATADIR
curl -L -o "$DATADIR/$BASENAME.wav" $URL
```

### Acoustic feature analysis

Now run GlottDNN Analysis program with default configuration
``` bash
Analysis "$DATADIR/$BASENAME.wav" ./config/config_default_16k.cfg
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
Synthesis "$DATADIR/$BASENAME" ./config/config_default_16k.cfg

# Move generated file
mv "$DATADIR/$BASENAME.syn.wav" "$DATADIR/$BASENAME.syn.sp.wav"    
```

A copy-synthesis wave file should now be at `./data/tmp/slt_arctic_a0001.syn.sp.wav`.
The single pulse excitation will sound somewhat buzzy, so let's try if we can do better.

### Single-file DNN training tutorial

The built-in PyTorch model can also be trained on one analyzed audio file.
This intentionally overfits the file and is a quick CPU smoke test rather than
a useful speech model.

First place one WAV file in the input directory and run:

```bash
mkdir -p data/single_file/wav
cp data/tmp/slt_arctic_a0001.wav data/single_file/wav/
glottdnn dnn_demo/config_single_file.yaml
```

The configuration uses
`data/single_file/slt_arctic_a0001.wav` and a small two-layer network. It
disables validation and test splits so all frames are used for training; when
no validation split exists, the trainer uses the training data for its
early-stopping metric. The generated DNN copy-synthesis output is written to
`data/single_file/syn/slt_arctic_a0001.syn.wav`.

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
Synthesis "$DATADIR/$BASENAME" ./config/config_default_16k.cfg $CONF_USR

# Move generated file
mv "$DATADIR/$BASENAME.syn.wav" "$DATADIR/$BASENAME.syn.paf.wav"       
```

Of course the original pulses are not available in many applications (such as text-to-speech). For this, we can use a trainable excitation model (neural net), which generates the pulses from acoustic features.

## Built-in neural net excitation model 

The present version uses PyTorch; all `theano` dependencies have been removed.

Note that the following is a toy example, since we now use only 10 audio files. This example is intended as a quick sanity check and can be easily run on a CPU. For more data and more complex models, a GPU is recommended.


Let's first download some data
```
sh ./dnn_demo/get_data.sh
```

Before we run anything, have a look into
```
./dnn_demo/config_dnn_demo.yaml
```

Then run the example script by saying
``` bash
glottdnn ./dnn_demo/config_dnn_demo.yaml
```

The demo script runs vocoder analysis, trains a DNN excitation model, and finally applies copy-synthesis to the samples.
After running, the copy-synthesis results are stored in `./dnn_demo/data/syn` and the original wave files are in `./dnn_demo/data/wav`.

### YAML configuration
Prepare a directory structure under and make file lists based on contents of the `wav` sub-directory
``` yaml
make_dirs: true
make_scp: true
```

Optionally, use REAPER for pitch (F0) and GCI analysis. 
Also optionally, use RAPT from SPTK for pitch analysis. These programs need to be installed separately, so this example does not use them. 

``` yaml
do_reaper_pitch_analysis: false
do_sptk_pitch_analysis: false
```

Use GlottDNN to extract glottal vocoder features and pulses for  excitation model training.
``` yaml
do_glott_vocoder_analysis: true
```

Package data and train an excitation model with the built-in PyTorch implementation.
``` yaml
make_dnn_train_data: true
make_dnn_infofile: true
do_dnn_training: true
```

Do copy synthesis (using the internal implementation of DNN excitation)
``` yaml
do_glott_vocoder_synthesis: true
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

## Python bindings

The CMake build also installs the `glottdnn_cpp` extension. The original
file-based workflows are available as:

```python
import glottdnn_cpp

glottdnn_cpp.analysis.run("input.wav", "config/config_default_16k.cfg")
glottdnn_cpp.synthesis.run("input", "config/config_default_16k.cfg")
```

Individual analysis stages can be called without parameter files:

```python
params = glottdnn_cpp.analysis.load_params("config/config_default_16k.cfg")
poly = glottdnn_cpp.analysis.spectral_analysis_with_params(
    signal, fundf, gci_indices, params
)
```

The `signal_processing` module contains NumPy wrappers for pure DSP functions.
The higher-level `vocoder` module provides an array-based analysis/synthesis
interface that returns Python dictionaries instead of intermediate parameter
files:

```python
from vocoder import analyze_file, synthesize

data = analyze_file("input.wav", "config/config_default_16k.cfg")
output = synthesize(data, "config/config_default_16k.cfg")
```

## Support

When in trouble, open an issue at GitHub. Others will likely have similar issues and it's best to solve them collectively

https://github.com/ljuvela/GlottDNN/issues

For questions, contact Lauri Juvela (lauri.juvela@aalto.fi) or Manu
Airaksinen (manu.airaksinen@aalto.fi).

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
