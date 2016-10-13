#!/bin/sh

# run analysis program
./src/Analysis ./wav/arctic_fe.wav ./config/config_default.cfg
/Users/ljuvela/git/GlottDNN/src/Synthesis ./wav/arctic_fe /Users/ljuvela/git/GlottDNN/test/config_dirwav