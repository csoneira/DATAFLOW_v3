# =============================================================================
# DATAFLOW_v3 Script Header v1
# Script: MASTER/STAGES/STAGE_0/REPROCESSING/UNPACKER_ZERO_STAGE_FILES/bin/unpack.sh
# Purpose: What to really change when the directory is changed:.
# Owner: DATAFLOW_v3 contributors
# Sign-off: csoneira <csoneira@ucm.es>
# Last Updated: 2026-03-02
# Runtime: bash
# Usage: bash MASTER/STAGES/STAGE_0/REPROCESSING/UNPACKER_ZERO_STAGE_FILES/bin/unpack.sh [options]
# Inputs: CLI args, config files, environment variables, and/or upstream files.
# Outputs: Files, logs, or process-level side effects.
# Notes: Keep behavior configuration-driven and reproducible.
# =============================================================================

# What to really change when the directory is changed:
#    unpack.sh, of course, the cd should lead to software
#    initConf.m, the HOME line, THAT MUST END WITH A SLASH
#    

cd $HOME/DATAFLOW_v3/MASTER/STAGES/STAGE_0/REPROCESSING/UNPACKER_ZERO_STAGE_FILES/software/ # <--------------------------------------------
octave --no-gui ./unpackingContinuous.m
