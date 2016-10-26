# run flags
make_dirs = True
make_scp = True
do_analysis = True
make_dnn_train_data = False
make_dnn_infofile = False
do_dnn_training = False

# directories
prjdir = '/Users/ljuvela/CODE/GlottDNN'
datadir = prjdir + '/data/slt16'

# general parameters
sampling_frequency = 16000
warping_lambda = 0.0

# programs
reaper = 'reaper'
sox = 'sox'
pitch = 'pitch -a 0 -s 16.0 -o 1 -p 80 -T 0.0 -L 50 -H 500 '
x2x = 'x2x'
Analysis = '/Users/ljuvela/CODE/GlottDNN/src/Analysis'
Synthesis = '/Users/ljuvela/CODE/GlottDNN/src/Synthesis'
config_default = '/Users/ljuvela/CODE/GlottDNN/config/config_default.cfg'

# nn input params
inputs = ['f0', 'gain', 'hnr', 'lsfg', 'lsf']
input_exts = ['.F0', '.Gain', '.HNR', '.LSFglot','.LSF']
input_dims = [1, 1, 5, 10, 30] # set feature to zero if not used
outputs = ['paf']
output_exts = ['.PAF']
output_dims = [400]

# dnn data conf
dnn_name = 'slt1'
train_data_dir = prjdir + '/nndata/traindata/' + dnn_name 
weights_data_dir = prjdir + '/nndata/weights/' + dnn_name
data_buffer_size = 100

#train_set = [1, 2 , 3, 4, 5]
train_set = [1]
val_set = [6]
test_set = [7]

# dnn train conf
n_hidden = [250, 100, 250]
learning_rate = 0.1
batch_size = 100
max_epochs = 10
