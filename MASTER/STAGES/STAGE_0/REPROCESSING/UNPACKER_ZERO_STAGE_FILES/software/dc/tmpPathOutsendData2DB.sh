# =============================================================================
# DATAFLOW_v3 Script Header v1
# Script: MASTER/STAGES/STAGE_0/REPROCESSING/UNPACKER_ZERO_STAGE_FILES/software/dc/tmpPathOutsendData2DB.sh
# Purpose: TmpPathOutsendData2DB.
# Owner: DATAFLOW_v3 contributors
# Sign-off: csoneira <csoneira@ucm.es>
# Last Updated: 2026-03-02
# Runtime: bash
# Usage: bash MASTER/STAGES/STAGE_0/REPROCESSING/UNPACKER_ZERO_STAGE_FILES/software/dc/tmpPathOutsendData2DB.sh [options]
# Inputs: CLI args, config files, environment variables, and/or upstream files.
# Outputs: Files, logs, or process-level side effects.
# Notes: Keep behavior configuration-driven and reproducible.
# =============================================================================

PGPASSWORD=pass psql -U user -p port -h 10.10.10.10  -d "systemName" -f - <<EOF

CREATE TEMPORARY TABLE tmp_table AS
