#!/usr/bin/env python3

'''
Docstring for INFERENCE_DICTIONARY_VALIDATION.STEP_3_SYNTHETIC_TIME_SERIES.STEP_3_2_SYNTHETIC_TIME_SERIES.synthetic_time_series

This code will take the time series created in the previous step and will, for each point in
the curve in the flux vs eff plane, weight the counts in the dictionary entries according to proximity to
each point in the curve. The idea in the end will be to have a table which looks like the data in 
/home/mingo/DATAFLOW_v3/INFERENCE_DICTIONARY_VALIDATION/STEP_1_SETUP/STEP_1_2_BUILD_DICTIONARY/OUTPUTS/FILES/dataset.csv
but actually its synthetic, it comes from the dictionary, and it has a well defined time (you have to make sure that the times
in the rows correspond to a time series). The output of this code will be the curve in te flux vs eff plane with
one point which will be enhanced for this plot showing with colours respect the points of the dictionary which % of each point its
data will be made of. The second plot will be the flux time series, the eff time series of the original curve and below the
global rate of the newly created table.
'''