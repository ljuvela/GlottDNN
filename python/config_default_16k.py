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
prjdir = '.' # change to your own local dir
datadir = os.path.join(prjdir, 'data', 'arctic_slt16') 

# general parameters
sampling_frequency = 16000
warping_lambda = 0.00
use_external_gci = False

# program calls, binaries assumed to be in $PATH, change to your own 
reaper = 'reaper' # optional, required if reaper is used for F0 or GCIs
sox = 'sox' 
pitch = 'pitch -a 0 -s 16.0 -p 80 '
x2x = 'x2x'
Analysis = prjdir + '/src/Analysis'
Synthesis = prjdir + '/src/Synthesis'
config_default = prjdir + '/config/config_default_16k.cfg'

# nn input params
inputs = ['f0', 'gain', 'hnr', 'slsf', 'lsf']
input_exts = ['.f0', '.gain', '.hnr', '.slsf','.lsf']
input_dims = [1, 1, 5, 10, 30] # set feature to zero if not used
outputs = ['pls']
output_exts = ['.pls']
output_dims = [400]

# dnn data conf
dnn_name = 'dnn_slt'
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

# import keras
make_data_provider = False
