# run flags
make_dirs = False
make_scp = False
do_analysis = True
make_dnn_train_data = False
make_dnn_infofile = False
do_dnn_training = False

# directories
prjdir = '/u/76/mairaksi/unix/Documents/CODE/GlottDNN'
datadir = prjdir + '/data/nancy48'

# general parameters
sampling_frequency = 48000
warping_lambda = 0.42

# programs
reaper = 'reaper'
sox = 'sox'
pitch = '/u/76/mairaksi/unix/Documents/SPTK-3.8/bin/pitch/pitch -a 0 -s 48.0 -o 1 -p 240 -T 0.0 -L 50 -H 500 '
x2x = '/u/76/mairaksi/unix/Documents/SPTK-3.8/bin/x2x/x2x'
Analysis = prjdir + '/src/Analysis'
Synthesis = prjdir + '/src/Synthesis'
config_default = prjdir + '/config/config_48.cfg'

# nn input params
inputs = ['f0', 'gain', 'hnr', 'lsfg', 'lsf']
input_exts = ['.F0', '.Gain', '.HNR', '.LSFglot','.LSF']
input_dims = [1, 1, 25, 10, 50] # set feature to zero if not used
outputs = ['paf']
output_exts = ['.PAF']
output_dims = [1600]

# dnn data conf
dnn_name = 'nancy48'
train_data_dir = prjdir + '/nndata/traindata/' + dnn_name 
weights_data_dir = prjdir + '/nndata/weights/' + dnn_name
data_buffer_size = 50

#train_set = [1, 2 , 3, 4, 5]
train_set = [2]
val_set = [6]
test_set = [7]

# dnn train conf
n_hidden = [250, 100, 250]
learning_rate = 0.1
batch_size = 100
max_epochs = 2000
