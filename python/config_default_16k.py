# run flags
make_dirs = True
make_scp = True
do_sptk_pitch_analysis = True
do_reaper_pitch_analysis = False
do_glott_vocoder_analysis = True
make_dnn_train_data = True
make_dnn_infofile = True
do_dnn_training = True
do_glott_vocoder_synthesis = True

# directories
prjdir = '/Users/ljuvela/CODE/GlottDNN' # add your own local install dir here
#datadir = prjdir + '/data/slt16'
datadir = prjdir + '/data/vowels'

# general parameters
sampling_frequency = 16000
warping_lambda = 0.00
use_external_gci = False


# programs
reaper = 'reaper'
sox = 'sox'
pitch = 'pitch -a 0 -s 16.0 -o 1 -p 80 -T 0.2 -L 50 -H 500 '
x2x = 'x2x'
Analysis = prjdir + '/src/Analysis'
Synthesis = prjdir + '/src/Synthesis'
config_default = prjdir + '/config/config_default_16k.cfg'

# nn input params
inputs = ['f0', 'gain', 'hnr', 'lsfg', 'lsf']
input_exts = ['.F0', '.Gain', '.HNR', '.LSFglot','.LSF']
input_dims = [1, 1, 0, 10, 30] # set feature to zero if not used
outputs = ['paf']
output_exts = ['.PAF']
output_dims = [400]

# dnn data conf
dnn_name = 'vowels'
train_data_dir = prjdir + '/nndata/traindata/' + dnn_name 
weights_data_dir = prjdir + '/nndata/weights/' + dnn_name
#data_buffer_size = 100
remove_unvoiced_frames = True
validation_ratio = 0.2
test_ratio = 0.1
max_number_of_files = 30


#train_set = [1, 2 , 3, 4, 5]
#train_set = [3,4,5,6,7,8]
#train_set = [1]
#val_set = [3]
#test_set = [4]

# dnn train conf
n_hidden = [150, 250, 300]
learning_rate = 0.01
batch_size = 100
max_epochs = 50
patience = 10  # early stopping criterion

# synthesis configs
use_dnn_generated_excitation = True
