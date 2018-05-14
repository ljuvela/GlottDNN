import numpy as np

# run flags
make_dirs = 1
make_scp = 1
do_reaper_pitch_analysis = 0
do_sptk_pitch_analysis = 0
do_glott_vocoder_analysis = 0
make_dnn_train_data = 0
make_dnn_infofile = 0
do_dnn_training = 0
do_glott_vocoder_synthesis = 0

make_data_provider = 1


# directories
prjdir = '/Users/ljuvela/CODE/GlottDNN' # add your own local install dir here
datadir = prjdir + '/data/nick'

# general parameters
sampling_frequency = 16000
warping_lambda = 0.00
use_external_gci = 0


# programs
reaper = '/usr/local/bin/reaper'
sox = '/usr/local/bin/sox'
pitch = '/usr/local/bin/pitch -a 0 -s 16.0 -o 1 -p 80 -T 0.2 -L 50 -H 800 '
x2x = '/usr/local/bin/x2x'
Analysis = prjdir + '/src/Analysis'
Synthesis = prjdir + '/src/Synthesis'
#config_default = prjdir + '/config/config_lauri_16k.cfg'
config_default = prjdir + '/config/config_default_16k.cfg'

# nn input params
#inputs = ['f0', 'gain', 'hnr', 'slsf', 'lsf']
#input_exts = ['.f0', '.gain', '.hnr', '.slsf','.lsf']
#input_dims = [1, 1, 5, 10, 30] # set feature to zero if not used
inputs = ['f0', 'mfcc']
input_exts = ['.f0', '.mfcc']
input_dims = [1, 20] # set feature to zero if not used

outputs = ['pls_mfcc']
output_exts = ['.pls']
output_dims = [400]

gen_out_dir = datadir + '/pls_gen'

data_type = np.float32

scalers = {'f0' : 'minmax',
           'gain' : 'minmax',
           'mfcc' : 'minmax',
           'hnr' : 'minmax',
           'slsf' : 'minmax',
           'lsf' : 'minmax',
           'pls_mfcc' : None,
           'pls' : None }

# dnn data conf
dnn_name = 'nick'
train_data_dir = prjdir + '/nndata/traindata/' + dnn_name 
weights_data_dir = prjdir + '/nndata/weights/' + dnn_name
remove_unvoiced_frames = True
validation_ratio = 0.2
test_ratio = 0.1
max_number_of_files = 30

# dnn train conf
n_hidden = [150, 250, 300]
learning_rate = 0.01
batch_size = 128
max_epochs = 50
patience = 10  # early stopping criterion

# synthesis configs
use_dnn_generated_excitation = False
