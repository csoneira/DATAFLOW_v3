# What to really change when the directory is changed:
#    unpack.sh, of course, the cd should lead to software
#    initConf.m, the HOME line, THAT MUST END WITH A SLASH
#    

cd $HOME/DATAFLOW_v3/MASTER/STAGE_0/UNPACKER_ZERO_STAGE_FILES/software/
octave --no-gui ./unpackingContinuous.m