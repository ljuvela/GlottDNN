#!/bin/sh

# run analysis program
./src/Analysis ./data/slt16/wav/arctic_fe.wav ./config/config_default.cfg
./src/Synthesis ./data/slt16/wav/arctic_fe.wav ./config/config_default.cfg