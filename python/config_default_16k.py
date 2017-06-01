# run flags
make_dirs = True
make_scp = False
do_reaper_pitch_analysis = False
do_sptk_pitch_analysis = False
do_glott_vocoder_analysis = False
make_dnn_train_data = False
make_dnn_infofile = False
do_dnn_training = False
do_glott_vocoder_synthesis = True

# directories
prjdir = '/l/CODE/GlottDNN' # add your own local install dir here
datadir = '/l/CODE/GlottDNN/data/iliad_mod'

# general parameters
sampling_frequency = 16000
warping_lambda = 0.0
use_external_gci = False


# programs
reaper = 'reaper'
sox = 'sox'
pitch = '/u/76/mairaksi/data/Documents/SPTK-3.8/bin/pitch/pitch -a 0 -s 16.0 -o 1 -p 80 -H 350 '
x2x = '/u/76/mairaksi/data/Documents/SPTK-3.8/bin/x2x/x2x'
Analysis = prjdir + '/src/Analysis'
Synthesis = prjdir + '/src/Synthesis'
config_default = prjdir + '/config/config_default_16k.cfg'

# nn input params
inputs = ['f0', 'gain', 'hnr', 'slsf', 'lsf']
input_exts = ['.F0', '.Gain', '.HNR', '.LSFglot','.LSF']
input_dims = [1, 1, 5, 10, 30] # set feature to zero if not used
outputs = ['paf']
output_exts = ['.paf']
output_dims = [400]

# dnn data conf
dnn_name = 'gdnn_jenny'
train_data_dir = prjdir + '/nndata/traindata/' + dnn_name 
weights_data_dir = '/work/t405/T40521/shared/vocomp/jenny/glottdnn/' + dnn_name
remove_unvoiced_frames = True
validation_ratio = 0.05
test_ratio = 0.05
max_number_of_files = 1000

# dnn train conf
n_hidden = [150, 250, 300]
learning_rate = 0.01
batch_size = 100
max_epochs = 50
patience = 10  # early stopping criterion

# synthesis configs
use_dnn_generated_excitation = True
