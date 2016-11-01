#!/usr/bin/python

from __future__ import division

import sys
import os
import numpy as np
import random
import wave
import math
import imp # for importing argv[1]


# Config file 
#import config as conf
if len(sys.argv) < 2:
    sys.exit("Usage: python GlottDnnScript.py config.py")
if os.path.isfile(sys.argv[1]):
   # conf = __import__(sys.argv[1])
    conf = imp.load_source('', sys.argv[1])
else:
    sys.exit("Config file " + sys.argv[1] + " does not exist")

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

def mkdir_p(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def make_directories():
    # Prepare environment 
    mkdir_p(conf.datadir + '/wav')
    mkdir_p(conf.datadir + '/raw')
    mkdir_p(conf.datadir + '/f0')
    mkdir_p(conf.datadir + '/gci')
    mkdir_p(conf.datadir + '/gain')
    mkdir_p(conf.datadir + '/lsf')
    mkdir_p(conf.datadir + '/lsfg')
    mkdir_p(conf.datadir + '/hnr')
    mkdir_p(conf.datadir + '/paf')
    mkdir_p(conf.datadir + '/scp')
    mkdir_p(conf.datadir + '/exc')
    mkdir_p(conf.datadir + '/syn')
    # Dnn directiories
    mkdir_p(conf.train_data_dir)
    mkdir_p(conf.weights_data_dir)
            
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
                    
    
def reaper_pitch_analysis():
    wavscp = conf.datadir + '/scp/wav.scp'
    with open(wavscp,'r') as wavfiles:
        for file in wavfiles:
            wavfile = file.rstrip()
            if os.path.isfile(wavfile):
                # define paths
                bname = os.path.splitext(os.path.basename(wavfile))[0]
                f0tmp1 = conf.datadir + '/f0/' + bname + '.F0tmp1'
                f0tmp2 = conf.datadir + '/f0/' + bname + '.F0tmp2'                
                f0file = conf.datadir + '/f0/' + bname + '.F0'
            
                gcitmp = conf.datadir + '/gci/' + bname + '.GCItmp'                
                gcifile = conf.datadir + '/gci/' + bname + '.GCI'
                
                # analysis commands
                cmd =  conf.reaper + ' -a -i ' + wavfile + ' -f ' + f0tmp1 + ' -p ' + gcitmp
                os.system(cmd)
                cmd = 'tail +8 ' + f0tmp1 + '| awk \'{print $3}\' | x2x +af | sopr -magic -1.0 -MAGIC 0.0  > ' + f0tmp2
                os.system(cmd)
                cmd = 'tail +8 ' + gcitmp + '| awk \'{print $1}\' | x2x +ad > ' + gcifile
                os.system(cmd)
            
                # read the file
                f0 = np.fromfile(f0tmp2, dtype=np.float32, count=-1, sep='')
                n_frames = len(f0)

                # calculate the sptk compatible number of frames 
                wave_read = wave.open(wavfile,'r')
                n_samples = wave_read.getnframes()
                sample_rate = wave_read.getframerate()
                n_frames_target =  int(math.ceil(n_samples/(0.005*sample_rate)))
                wave_read.close
 
                # zero pad and save f0
                f0_true = np.zeros(n_frames_target, dtype= np.float32)
                npad_start = 2
                f0_true[npad_start:npad_start+n_frames] = f0
                f0_true.astype(np.float64).tofile(f0file, sep='',format="%f")

                # remove tmp
                os.remove(f0tmp1)
                os.remove(f0tmp2)
                os.remove(gcitmp)

def glott_vocoder_analysis():
    wavscp = conf.datadir + '/scp/wav.scp'
    with open(wavscp,'r') as wavfiles:
        for file in wavfiles:
            wavfile = file.rstrip()
            if os.path.isfile(wavfile):
                bname = os.path.splitext(os.path.basename(wavfile))[0]
                f0file = conf.datadir + '/f0/' + bname + '.F0'
                gcifile = conf.datadir + '/gci/' + bname + '.GCI'
                config_user = 'config_user.cfg'
                conf_file = open(config_user,'w');
                if conf.do_sptk_pitch_analysis or do_reaper_pitch_analysis:
                    conf_file.write('USE_EXTERNAL_F0 = true;\n')
                    conf_file.write('EXTERNAL_F0_FILENAME = \"' + f0file + '\";\n' )
                else:
                    conf_file.write('USE_EXTERNAL_F0 = false;\n')
                if conf.use_external_gci:
                    conf_file.write('USE_EXTERNAL_GCI = true;\n')
                    conf_file.write('EXTERNAL_GCI_FILENAME = \"' + gcifile + '\";\n' )
                conf_file.write('SAMPLING_FREQUENCY = ' + str(conf.sampling_frequency) +';\n')
                conf_file.write('WARPING_LAMBDA_VT = '+ str(conf.warping_lambda) +';\n')
                conf_file.write('DATA_DIRECTORY = \"' + conf.datadir + '\";\n')
                conf_file.close()
                cmd = conf.Analysis + ' ' + wavfile + ' ' + conf.config_default + ' ' + config_user
                os.system(cmd)

def glott_vocoder_synthesis():
    wavscp = conf.datadir + '/scp/wav.scp'
    with open(wavscp,'r') as wavfiles:
        for file in wavfiles:
            wavfile = file.rstrip()
            if os.path.isfile(wavfile):
                bname = os.path.splitext(os.path.basename(wavfile))[0]
                f0file = conf.datadir + '/f0/' + bname + '.F0'
                config_user = 'config_user.cfg'
                conf_file = open(config_user,'w');
                conf_file.write('SAMPLING_FREQUENCY = ' + str(conf.sampling_frequency) +';\n')
                conf_file.write('WARPING_LAMBDA_VT = '+ str(conf.warping_lambda) +';\n')
                conf_file.write('DATA_DIRECTORY = \"' + conf.datadir + '\";\n')
                conf_file.close()
                cmd = conf.Synthesis + ' ' + wavfile + ' ' + conf.config_default + ' ' + config_user
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
            # remove unvoiced frames if requested
            if conf.remove_unvoiced_frames:
                input_data = input_data[input_data[:,0] > 0,:]
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

            # read output data
            feat_start = 0
            for (ftype, ext, dim) in zip( conf.outputs, conf.output_exts, conf.output_dims): 
                if dim > 0:
                    feat_file = conf.datadir + '/'+ ftype + '/' + bname + ext
                    feat = np.fromfile(feat_file, dtype=np.float64, count=-1, sep='')
                    feat = np.reshape(feat, (-1,dim))
                    output_data[:,feat_start:feat_start+dim ] = feat
                    feat_start += dim
            
            
            # remove unvoiced frames if requested
            if conf.remove_unvoiced_frames:
                output_data = output_data[input_data[:,0] > 0,:]
                input_data = input_data[input_data[:,0] > 0,:]
            
            # normalize and write input data
            input_data = (input_data - in_min) / (in_max - in_min) * (new_max - new_min) + new_min
            input_data.astype(np.float32).tofile(in_fid, sep='',format="%f")
            
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
    if conf.do_reaper_pitch_analysis:
        reaper_pitch_analysis()
       
    # SPTK pitch estimation
    if conf.do_sptk_pitch_analysis:
        sptk_pitch_analysis()
    
    # GlottDNN Analysis
    if conf.do_glott_vocoder_analysis:
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

    # Copy synthesis
    if conf.do_glott_vocoder_synthesis:
        glott_vocoder_synthesis()
    
if __name__ == "__main__":
    main(sys.argv[1:])
