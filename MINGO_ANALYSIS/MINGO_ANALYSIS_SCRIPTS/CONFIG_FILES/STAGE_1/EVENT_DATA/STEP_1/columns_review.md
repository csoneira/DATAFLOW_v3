# Columns in pipeline files

## TASK_0 -> TASK_1
### from the ~/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/TASK_1/INPUT_FILES/COMPLETED_DIRECTORY/raw_*.parquet

acquisition_type
datetime
event_id
p1_s1_eb_q
p1_s1_eb_t
p1_s1_ef_q
p1_s1_ef_t
p1_s2_eb_q
p1_s2_eb_t
p1_s2_ef_q
p1_s2_ef_t
p1_s3_eb_q
p1_s3_eb_t
p1_s3_ef_q
p1_s3_ef_t
p1_s4_eb_q
p1_s4_eb_t
p1_s4_ef_q
p1_s4_ef_t
p2_s1_eb_q
p2_s1_eb_t
p2_s1_ef_q
p2_s1_ef_t
p2_s2_eb_q
p2_s2_eb_t
p2_s2_ef_q
p2_s2_ef_t
p2_s3_eb_q
p2_s3_eb_t
p2_s3_ef_q
p2_s3_ef_t
p2_s4_eb_q
p2_s4_eb_t
p2_s4_ef_q
p2_s4_ef_t
p3_s1_eb_q
p3_s1_eb_t
p3_s1_ef_q
p3_s1_ef_t
p3_s2_eb_q
p3_s2_eb_t
p3_s2_ef_q
p3_s2_ef_t
p3_s3_eb_q
p3_s3_eb_t
p3_s3_ef_q
p3_s3_ef_t
p3_s4_eb_q
p3_s4_eb_t
p3_s4_ef_q
p3_s4_ef_t
p4_s1_eb_q
p4_s1_eb_t
p4_s1_ef_q
p4_s1_ef_t
p4_s2_eb_q
p4_s2_eb_t
p4_s2_ef_q
p4_s2_ef_t
p4_s3_eb_q
p4_s3_eb_t
p4_s3_ef_q
p4_s3_ef_t
p4_s4_eb_q
p4_s4_eb_t
p4_s4_ef_q
p4_s4_ef_t
param_hash
topology_task1_channel
transferred_task0_acq_to_raw
tt_task0_acq
tt_task0_raw


## TASK_0 -> SELFTRIGGER
### from the ~/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/TASK_0/OUTPUT_FILES/selftrigger_raw_*.parquet

No parquet files found


## TASK_1 -> TASK_2
### from the ~/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/TASK_2/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet

acquisition_type
datetime
event_id
filter_task1_datetime_in_range_pass
filter_task1_problematic_channel_count
filter_task1_problematic_channel_exact
filter_task1_tt_task1_clean_pass
p1_s1_eb_q
p1_s1_eb_t
p1_s1_ef_q
p1_s1_ef_t
p1_s2_eb_q
p1_s2_eb_t
p1_s2_ef_q
p1_s2_ef_t
p1_s3_eb_q
p1_s3_eb_t
p1_s3_ef_q
p1_s3_ef_t
p1_s4_eb_q
p1_s4_eb_t
p1_s4_ef_q
p1_s4_ef_t
p2_s1_eb_q
p2_s1_eb_t
p2_s1_ef_q
p2_s1_ef_t
p2_s2_eb_q
p2_s2_eb_t
p2_s2_ef_q
p2_s2_ef_t
p2_s3_eb_q
p2_s3_eb_t
p2_s3_ef_q
p2_s3_ef_t
p2_s4_eb_q
p2_s4_eb_t
p2_s4_ef_q
p2_s4_ef_t
p3_s1_eb_q
p3_s1_eb_t
p3_s1_ef_q
p3_s1_ef_t
p3_s2_eb_q
p3_s2_eb_t
p3_s2_ef_q
p3_s2_ef_t
p3_s3_eb_q
p3_s3_eb_t
p3_s3_ef_q
p3_s3_ef_t
p3_s4_eb_q
p3_s4_eb_t
p3_s4_ef_q
p3_s4_ef_t
p4_s1_eb_q
p4_s1_eb_t
p4_s1_ef_q
p4_s1_ef_t
p4_s2_eb_q
p4_s2_eb_t
p4_s2_ef_q
p4_s2_ef_t
p4_s3_eb_q
p4_s3_eb_t
p4_s3_ef_q
p4_s3_ef_t
p4_s4_eb_q
p4_s4_eb_t
p4_s4_ef_q
p4_s4_ef_t
param_hash
passes_task_1
topology_task1_channel
transferred_task0_acq_to_raw
transferred_task1_raw_to_clean
tt_task0_acq
tt_task0_raw
tt_task1_clean


## TASK_2 -> TASK_3
### from the ~/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/TASK_3/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet

No parquet files found


## TASK_3 -> TASK_4
### from the ~/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/TASK_4/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet

No parquet files found


## TASK_4 -> TASK_5
### from the ~/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/MINGO00/STAGE_1/EVENT_DATA/STEP_1/TASK_5/INPUT_FILES/COMPLETED_DIRECTORY/*_*.parquet

No parquet files found


## TASK_5 -> OUT
### from the ~/DATAFLOW_v3/MINGO_ANALYSIS/MINGO_ANALYSIS_STATIONS/MINGO04/STAGE_1_PRODUCTS/EVENT_DATA/PARQUET_LAKE/postprocessed_*.parquet

No parquet files found

