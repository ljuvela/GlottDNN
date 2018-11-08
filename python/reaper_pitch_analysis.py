#!/usr/bin/python

# Python wrapper for Reaper F0 and GCI estimation
# Lauri Juvela, lauri.juvela@aalto.fi

import sys
import os
import numpy as np
import wave
import argparse


def estimate(wavfile, f0file, gcifile, reaper_path, two_stage_estimation=False):

    # check that input files exist
    if not os.path.isfile(wavfile):
        print('Error: input file ' + wavfile + ' not found')
        return

    f0tmp = f0file + '.tmp'
    gcitmp = gcifile + '.tmp'

    frame_step = 0.005  # can be given as argument
    #frame_size = 0.020 # checked from reaper source code, not an argument

    # first round estimate with wide range
    cmd = reaper_path + ' -a -i ' + wavfile + ' -f ' + \
        f0tmp + ' -p ' + gcitmp + ' -x 500 -m 40' + ' -e ' + str(frame_step)
    os.system(cmd)

    if two_stage_estimation:
        f0_mat = np.loadtxt(f0tmp, skiprows=7, dtype=np.float32)
        vuv = f0_mat[:, 1]
        f0 = f0_mat[:, 2]
        f0_voiced = f0[vuv > 0.0]

        # estimate new search range
        lf0 = np.log(f0_voiced)
        m = np.mean(lf0)
        s = np.std(lf0)
        lf0_max = m + 3*s
        lf0_min = m - 3*s
        f0_max = np.exp(lf0_max)
        f0_min = np.exp(lf0_min)

        #n_out_of_range = sum((f0_voiced < f0_min) + (f0_voiced > f0_max))

        f0_max = "{:.2f}".format(f0_max)
        f0_min = "{:.2f}".format(f0_min)
        f0_min = '40' # override low limit

        # second round estimate with narrow range
        print("F0 search range {} - {}".format(f0_min, f0_max))
        cmd = reaper_path + ' -a -i ' + wavfile + ' -f ' + \
            f0tmp + ' -p ' + gcitmp + ' -x ' + f0_max + ' -m ' + f0_min \
            + ' -e ' + str(frame_step)
        os.system(cmd)

    f0_mat = np.loadtxt(f0tmp, skiprows=7, dtype=np.float32)
    vuv = f0_mat[:, 1]
    f0 = f0_mat[:, 2]
    f0_voiced = f0[vuv > 0.0]

    # zeros mark unvoiced
    f0[vuv == 0.0] = 0.0

    # only take voiced pitch marks
    gci_mat = np.loadtxt(gcitmp, skiprows=7, dtype=np.float32)
    gci_voiced = gci_mat[:, 1] > 0.0
    gci = gci_mat[gci_voiced, 0]
    gci.astype(np.float32).tofile(gcifile)

    # calculate the number of frames sptk would give
    wave_read = wave.open(wavfile, 'r')
    n_samples = wave_read.getnframes()
    sample_rate = wave_read.getframerate()
    wave_read.close()
    n_frames_target = int(np.ceil(n_samples/(0.005*sample_rate)))
    n_frames = len(f0)

    # zero pad and save f0
    f0_true = np.zeros(n_frames_target, dtype=np.float32)
    npad_start = 2  # depends on framing, by default reaper analysis window 20ms and hop size 5ms result in inital shift of two
    f0_true[npad_start:npad_start+n_frames] = f0
    f0_true.astype(np.float32).tofile(f0file)

    # remove tmp
    os.remove(f0tmp)
    os.remove(gcitmp)



def get_args():

   parser = argparse.ArgumentParser(
       description='Wrapper for reaper pitch extraction')
   parser.add_argument(
       '--reaper_path', help='Path to REAPER binary', type=str, required=True)
   parser.add_argument('--wav_file', help='Input wav file',
                       type=str, required=True)
   parser.add_argument('--gci_file', help='Output GCI file',
                       type=str, required=True)
   parser.add_argument('--f0_file', help='Output F0 file',
                       type=str, required=True)
   args = parser.parse_args()
   return args




# main function
if __name__ == "__main__":
   args = get_args()
   estimate(wavfile=args.wav_file, f0file=args.f0_file,
            gcifile=args.gci_file, reaper_path=args.reaper_path)
