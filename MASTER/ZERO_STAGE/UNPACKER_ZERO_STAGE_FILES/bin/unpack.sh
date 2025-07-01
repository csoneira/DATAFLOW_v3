# What to really change when the directory is changed:
#    unpack.sh, of course, the cd should lead to software
#    initConf.m, the HOME line, THAT MUST END WITH A SLASH
#    

cd /home/mingo/DATAFLOW_v3/MASTER/ZERO_STAGE/UNPACKER_ZERO_STAGE_FILES/software/
octave --no-gui ./unpackingContinuous.m