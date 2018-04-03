# Vocoder configuration 

## Analysis and Synthesis directories
- `DATA_DIRECTORY` : Root directory for speech data for analysis. Place wave files in `wav` subdirectory
- `SAVE_TO_DATADIR_ROOT` = If `true`, all parameters are written and read in the same directory as where the wave file is 

## General shared parameters
- `SAMPLING_FREQUENCY` : 
    Sampling frequency should match that of the wav file
- `FRAME_LENGTH` :
    Analysis frame length (in ms)
- `UNVOICED_FRAME_LENGTH` :
    Analysis frame length in unvoiced frames. Shorter frames can better capture plosives and other impulse-like unvoiced events.
- `F0_FRAME_LENGTH` : Frame length used for fundamental frequency analysis. 
- `FRAME_SHIFT` :
    Frame step length (in ms)
- `LPC_ORDER_VT` : 
    LPC order for the vocal tract filter
- `LPC_ORDER_GLOT` :
    LPC order for the glottal source
- `HNR_ORDER` :
    Number of ERB bands for Harmonic-to-noise ratio
- `DATA_TYPE` : Data type for saving and reading parameters. Valid types are "ASCII" / "DOUBLE" / "FLOAT"    

## General analysis parameters:
- `SIGNAL_POLARITY` : Signal polarity heavily affects glottal closure instant detection. If you know the signal polarity to be positive, use `"DEFAULT"` or if negative use  `"INVERT"`. If you are unsure, use `"DETECT"`.
- `HP_FILTERING`: Use high pass filter at 50Hz to prevent low frequency rumble  

## Parameters for F0 estimation:
- `F0_MIN` Minimum allowed F0 value in Hz
- `F0_MAX` Maximum allowed F0 value in Hz
- `VOICING_THRESHOLD` : Threshold value for voicing decision based on low frequency band energy compared to high bands
- `ZCR_THRESHOLD`: Zero crossing rate threshold for voicing decision
- `RELATIVE_F0_THRESHOLD`
- `F0_CHECK_RANGE`: Number of channels for dynamic programming to prevent octave jumps 

## Use of external F0 and GCI estimators	
- `USE_EXTERNAL_F0`: Use external F0 estimate
- `EXTERNAL_F0_FILENAME` =    "data/nancy/BC2011_nancy_ARC_002.F0_sptk"; # Same format as DATA_TYPE expected (ascii/double/float)
- `USE_EXTERNAL_GCI` Use external estimator for glottal closure instants (GCIs). (REAPER is recommended)
- `EXTERNAL_GCI_FILENAME` =   "wav/arctic_fe.GCIrea";  # Format: Each row has one GCI's timing (in seconds), saved as DATA_TYPE
- `USE_EXTERNAL_LSF_VT` Uses external vocal tract LSF file for inverse filtering (order must match with the config).
- `EXTERNAL_LSF_VT_FILENAME` Filename as string.


## Pulses as features (PAF): Parameters for extracting pulses and synthesis:
- `MAX_PULSE_LEN_DIFF`:
    Percentage of how much pulse length can differ from F0. Pulses are searched iteratively until the nearest pulse fulfilling the length condition is found.     
- `PAF_PULSE_LENGTH` :
    Pulses-as-features length in samples. If interpolation is not used, this should be large enough to fit two pitch periods at the lowest F0.
- `USE_PULSE_INTERPOLATION` :
    If `true`, two pitch-period pulses are interpolated to fill the feature vector. If `false`, the pulse is only centered at GCI.
- `USE_WAVEFORMS_DIRECTLY` :
    If `true`, the speech waveform is extracted directly instead of the inverse filtered waveform.
- `PAF_WINDOW` :
    Select the windowing function applied to the pulse at analysis. Valid options are 
    `"NONE"`/`"HANN"`/`"COSINE"`/`"KBD"`
- `USE_PAF_ENERGY_NORM` :
    Normalize the pulse to unit energy. May induce amplitude modulation artefacts in synthesis.

## Parameters for spectral modeling and glottal inverse filtering (GIF):

Template settings for established GIF methods:
- IAIF: `USE_ITERATIVE_GIF` = true; `LP_WEIGHTING` = "NONE"; `WARPING_VT` = 0.0;
- QCP: `USE_ITERATIVE_GIF` = false; `LP_WEIGHTING` = "AME"; WARPING_VT = 0.0;

- `USE_ITERATIVE_GIF`
- `USE_PITCH_SYNCHRONOUS_ANALYSIS`
- `LPC_ORDER_GLOT_IAIF`: Order of the LPC analysis for voice source in IAIF
- `LP_WEIGHTING_FUNCTION`: Weighting function for weighted linear predictive analysis. Select between `"NONE"` / `"AME"` / `"STE"`. Attenuated main excitation (AME) corresponds to QCP analysis.
- `AME_DURATION_QUOTIENT`
- `AME_POSITION_QUOTIENT`
- `GIF_PRE_EMPHASIS_COEFFICIENT`: First order pre-emphasis filter coefficient for GIF
- `WARPING_LAMBDA_VT`: Bi-linear frequency warping coefficient (not used with QMF).
QMF sub-band analysis (for full-band speech)
- `QMF_SUBBAND_ANALYSIS` = Use quadrature mirror filter (QMF) band splitting for analysis. Always uses QCP for low-band and LPC for high-band, ignores warping
- `LPC_ORDER_QMF1`: Low-band linear predictor order for QMF 
- `LPC_ORDER_QMF2`: High-band linear predictor order for QMF 

## Select parameters to be extracted to files:
- `EXTRACT_F0`:     
- `EXTRACT_GAIN`:      
- `EXTRACT_LSF_VT`:   
- `EXTRACT_LSF_GLOT`:
- `EXTRACT_HNR`:
- `EXTRACT_GLOTTAL_EXCITATION`: Save full length estimated glottal excitation signal
- `EXTRACT_GCI_SIGNAL`:
- `EXTRACT_PULSES_AS_FEATURES`: 

## Synthesis: General parameters:
- `USE_GENERIC_ENVELOPE`: Read a full resolution magnitude spectrum and use its minimum phase version as vocal tract filter
- `USE_SPECTRAL_MATCHING`: Use spectral matching for excitation
- `PSOLA_WINDOW`: Window type used for pitch-synchronous overlap add. Must be compatible with `PAF_WINDOW`. Select between `"NONE"` (Rectangular over full frame) /`"HANN"`/`"COSINE"`/`"KBD"`
- `EXCITATION_METHOD` Select between
    - `"SINGLE_PULSE"` Uses a fixed glottal excitation pulse which is modified in accordance with acoustic parameters.
    - `"DNN_GENERATED"` Uses internal implementation of feedforward DNN for predicting glottal pulse shape from acoustic features.
    - `"PULSES_AS_FEATURES"`
- `USE_ORIGINAL_EXCITATION` =	 false;
- `USE_PAF_UNVOICED` =      	 false;
- `USE_WSOLA` =		 true;