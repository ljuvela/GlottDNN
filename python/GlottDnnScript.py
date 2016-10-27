#!/usr/bin/python 
import sys
import os
import numpy as np
import random

# Config file 
import config as conf

# Import Theano-based DNN training if used 
if conf.do_dnn_training:
    import TrainDnn

def write_dnn_infofile():
    file = open(conf.weights_data_dir + '/' + conf.dnn_name + '.dnnInfo', 'w')
    # write layer sizes
    n_input = sum(conf.input_dims)
    n_output = sum(conf.output_dims)
    file.write('LAYERS = [')
    file.write(str(n_input) + ', ')
    for n_hidden in conf.n_hidden:
        file.write(str(n_hidden) + ', ')
    file.write(str(n_output))
    file.write('];\n')
    # write activations
    n_input = sum(conf.input_dims)
    n_output = sum(conf.output_dims)
    file.write('ACTIVATIONS = [')
    #file.write(str(n_input) + ', ')
    for n_hidden in conf.n_hidden:
        file.write( '\"S\", ')
    file.write('\"L\", \"L\"')
    file.write('];\n')    
    # write feature orders 
    file.write('F0_ORDER = ' + str(conf.input_dims[0]) + ';\n')
    file.write('GAIN_ORDER = ' + str(conf.input_dims[1]) + ';\n')
    file.write('HNR_ORDER = ' + str(conf.input_dims[2]) + ';\n')
    file.write('LPC_ORDER_GLOT = ' + str(conf.input_dims[3]) + ';\n')
    file.write('LPC_ORDER_VT = ' + str(conf.input_dims[4]) + ';\n')
    file.write('SAMPLING_FREQUENCY = ' + str(conf.sampling_frequency) + ';\n')
    file.write('WARPING_LAMBDA_VT = ' + str(conf.warping_lambda) + ';\n')
    file.close()


def make_directories():
    # Prepare environment 
    dirpath = conf.datadir + '/wav'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dirpath = conf.datadir + '/raw'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)    
    dirpath = conf.datadir + '/f0'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dirpath = conf.datadir + '/gain'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dirpath = conf.datadir + '/lsf'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dirpath = conf.datadir + '/lsfg'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dirpath = conf.datadir + '/hnr'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dirpath = conf.datadir + '/paf'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)        
    dirpath = conf.datadir + '/scp'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dirpath = conf.datadir + '/exc'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    dirpath = conf.datadir + '/syn'
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    # Dnn directiories
    if not os.path.exists(conf.train_data_dir):
        os.makedirs(conf.train_data_dir)
    if not os.path.exists(conf.weights_data_dir):
        os.makedirs(conf.weights_data_dir)

            
def make_file_lists():
    scpfile = open(conf.datadir + '/scp/wav.scp','w') 
    for f in sorted(set(os.listdir(conf.datadir + '/wav'))):
        if f.endswith(".wav"):
            scpfile.write(os.path.abspath(conf.datadir + '/wav/' + f + '\n'))
    scpfile.close()

def sptk_pitch_analysis():
        wavscp = conf.datadir + '/scp/wav.scp'
        with open(wavscp,'r') as wavfiles:
            for file in wavfiles:
                wavfile = file.rstrip()
                if os.path.isfile(wavfile):
                    bname = os.path.splitext(os.path.basename(wavfile))[0]
                    # convert to .raw
                    rawfile = conf.datadir + '/raw/' + bname + '.raw'
                    f0file = conf.datadir + '/f0/' + bname + '.F0'
                    cmd = conf.sox + ' ' + wavfile + ' ' + rawfile
                    os.system(cmd)
                    # sptk pitch estimation 
                    cmd = conf.x2x + ' +sf ' + rawfile + '|' \
                        + conf.pitch + ' | ' + conf.x2x + ' +fd >' + f0file
                    os.system(cmd)
    
def glott_vocoder_analysis():
    wavscp = conf.datadir + '/scp/wav.scp'
    with open(wavscp,'r') as wavfiles:
        for file in wavfiles:
            wavfile = file.rstrip()
            if os.path.isfile(wavfile):
                bname = os.path.splitext(os.path.basename(wavfile))[0]
                f0file = conf.datadir + '/f0/' + bname + '.F0'
                config_user = 'config_user.cfg'
                conf_file = open(config_user,'w');
                conf_file.write('USE_EXTERNAL_F0 = true;\n')
                conf_file.write('EXTERNAL_F0_FILENAME = \"' + f0file + '\";\n' )
                conf_file.write('SAMPLING_FREQUENCY = ' + str(conf.sampling_frequency) +';\n')
                conf_file.write('WARPING_LAMBDA_VT = '+ str(conf.warping_lambda) +';\n')
                conf_file.write('DATA_DIRECTORY = \"' + conf.datadir + '\";\n')
                conf_file.close()
                cmd = conf.Analysis + ' ' + wavfile + ' ' + conf.config_default + ' ' + config_user
                os.system(cmd)

def package_data():
    # read and shuffle wav filelist
    wavscp = conf.datadir + '/scp/wav.scp' 
    with open(wavscp,'r') as wavfiles:
        filelist = wavfiles.read().splitlines()    
    random.shuffle(filelist)
    
    # initialize global min and max
    in_min = 9999*np.ones([1,sum(conf.input_dims)],dtype=np.float64)
    in_max = -9999*np.ones([1,sum(conf.input_dims)],dtype=np.float64)    
    
    n_frames = np.zeros([len(filelist)], dtype='int')
    for file_idx, wavfile in enumerate(filelist):
        if os.path.isfile(wavfile):
            bname = os.path.splitext(os.path.basename(wavfile))[0]
                # todo: save n_frames
            f0_file = conf.datadir + '/f0/' + bname + '.F0' 
            n_frames[file_idx] = (np.fromfile(f0_file, dtype=np.float64, count=-1, sep='')).shape[0]
            # allocate file data
            input_data = np.empty([n_frames[file_idx], sum(conf.input_dims)], dtype=np.float64)
            feat_start = 0
            for (ftype, ext, dim) in zip( conf.inputs, conf.input_exts, conf.input_dims):
                if dim > 0:
                    # read feat  data
                    feat_file = conf.datadir + '/'+ ftype + '/' + bname + ext
                    feat = np.fromfile(feat_file, dtype=np.float64, count=-1, sep='')
                    # reshape
                    feat = np.reshape(feat, (-1,dim))
                    # set to input data matrix
                    input_data[:,feat_start:feat_start+dim ] = feat
                    feat_start += dim
            # update global min and max
            in_min = np.minimum(np.amin(input_data, axis=0), in_min)
            in_max = np.maximum(np.amax(input_data, axis=0), in_max)

    new_min = 0.1
    new_max = 0.9
    
    batch_index = 1
    in_fid = open(conf.train_data_dir + '/' + conf.dnn_name + '.' + str(batch_index) + '.idat' ,'w')
    out_fid = open(conf.train_data_dir + '/' + conf.dnn_name + '.' + str(batch_index) + '.odat' ,'w')
            
    for file_idx, wavfile in enumerate(filelist):
        if file_idx > 0 and file_idx % conf.data_buffer_size == 0:
            in_fid.close()
            out_fid.close()
            batch_index += 1
            in_fid = open(conf.train_data_dir + '/' + conf.dnn_name + '.' + str(batch_index) + '.idat' ,'w')
            out_fid = open(conf.train_data_dir + '/' + conf.dnn_name + '.' + str(batch_index) + '.odat' ,'w')
            
   
        if os.path.isfile(wavfile):
            bname = os.path.splitext(os.path.basename(wavfile))[0]                
            # allocate input and output data
            input_data = np.empty([n_frames[file_idx], sum(conf.input_dims)], dtype=np.float64)
            output_data = np.empty([n_frames[file_idx], sum(conf.output_dims)], dtype=np.float64)
   
            # read input data
            feat_start = 0
            for (ftype, ext, dim) in zip( conf.inputs, conf.input_exts, conf.input_dims):
                if dim > 0:
                    feat_file = conf.datadir + '/'+ ftype + '/' + bname + ext
                    feat = np.fromfile(feat_file, dtype=np.float64, count=-1, sep='')
                    feat = np.reshape(feat, (-1,dim))
                    input_data[:,feat_start:feat_start+dim ] = feat
                    feat_start += dim

            # normalize and write input data
            input_data = (input_data - in_min) / (in_max - in_min) * (new_max - new_min) + new_min
            input_data.astype(np.float32).tofile(in_fid, sep='',format="%f")
           
            # read output data
            feat_start = 0
            for (ftype, ext, dim) in zip( conf.outputs, conf.output_exts, conf.output_dims): 
                if dim > 0:
                    feat_file = conf.datadir + '/'+ ftype + '/' + bname + ext
                    feat = np.fromfile(feat_file, dtype=np.float64, count=-1, sep='')
                    feat = np.reshape(feat, (-1,dim))
                    output_data[:,feat_start:feat_start+dim ] = feat
                    feat_start += dim

            # write output data
            output_data.astype(np.float32).tofile(out_fid, sep='',format="%f")

    # close files
    in_fid.close()
    out_fid.close()

    # write input min and max
    fid = open(conf.weights_data_dir + '/' + conf.dnn_name + '.dnnMinMax' ,'w')
    in_min.astype(np.float64).tofile(fid, sep='', format="%f")
    in_max.astype(np.float64).tofile(fid, sep='', format="%f")
    fid.close()
    

def main(argv):
   
    # Make directories
    if conf.make_dirs:
        make_directories()

    # Make SCP list for wav
    if conf.make_scp:
        make_file_lists()

    # Reaper pitch estimation
       # TODO
       
    # SPTK pitch estimation
    if conf.do_analysis:
        sptk_pitch_analysis()
    
    # GlottDNN Analysis
    if conf.do_analysis:
        glott_vocoder_analysis()

    # Package data for Theano
    if conf.make_dnn_train_data:
        package_data()

    # Write Dnn infofile
    if conf.make_dnn_infofile:
        write_dnn_infofile()

    # Train Dnn with Theano
    if conf.do_dnn_training:
        dim_in = sum(conf.input_dims)
        dim_out = sum(conf.output_dims)
        TrainDnn.evaluate_dnn(n_in=dim_in, n_out=dim_out, n_hidden=conf.n_hidden, batch_size=conf.batch_size, 
                 learning_rate=conf.learning_rate, n_epochs = conf.max_epochs)

if __name__ == "__main__":
    main(sys.argv[1:])
