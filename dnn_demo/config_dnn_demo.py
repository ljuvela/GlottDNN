import os

# run flags
make_dirs = 1
make_scp = 1
do_reaper_pitch_analysis = 0
do_sptk_pitch_analysis = 0
do_glott_vocoder_analysis = 1
make_dnn_train_data = 1
make_dnn_infofile = 1
do_dnn_training = 1
do_glott_vocoder_synthesis = 1

# directories
prjdir = './' # change to your own local dir
datadir = os.path.join(prjdir, 'dnn_demo', 'data') 

# path to REAPER binary (optional)
reaper = 'reaper' 
# paths to SPTK binaries (optional)
sox = 'sox' 
pitch = 'pitch -a 0 -s 16.0 -p 80 '
x2x = 'x2x'

# GlottDNN binaries and default config
Analysis = prjdir + '/src/Analysis'
Synthesis = prjdir + '/src/Synthesis'
config_default = prjdir + '/dnn_demo/config_dnn_demo.cfg'

# general parameters
sampling_frequency = 16000
warping_lambda = 0.00
use_external_gci = False

# Neural net input params
inputs = ['f0', 'gain', 'hnr', 'slsf', 'lsf']
input_exts = ['.f0', '.gain', '.hnr', '.slsf','.lsf']
input_dims = [1, 1, 5, 10, 30] # set feature to zero if not used
outputs = ['pls']
output_exts = ['.pls']
output_dims = [400]

# dnn data conf
dnn_name = 'dnn_demo_slt'
train_data_dir = os.path.join(prjdir, 'nndata/traindata', dnn_name) 
weights_data_dir = os.path.join(prjdir, 'nndata/weights', dnn_name)
remove_unvoiced_frames = True
validation_ratio = 0.10
test_ratio = 0.10
max_number_of_files = 1000

# dnn train conf
n_hidden = [150, 250, 300]
learning_rate = 0.01
batch_size = 100
max_epochs = 50
patience = 5  # early stopping criterion

# synthesis configs
use_dnn_generated_excitation = True