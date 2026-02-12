#!/usr/bin/env python3

'''
Docstring for INFERENCE_DICTIONARY_VALIDATION.STEP_3_SYNTHETIC_TIME_SERIES.STEP_3_1_TIME_SERIES_CREATION.create_time_series

This script takes from the config file limits in the flux_cm2_min and eff and
will create a time series of the flux_cm2_min and eff values for the period. In principle the time series will be a smooth path
in the flux vs eff plane, it might be of constant flux and varying eff, it might be of constant eff and varying flux,
both might vary so the line follows a constant global rate line... I want to be able to generate several different curves in
the flux vs eff plane. In the beginning this curve won't have "time" associated, then in the config file a length of a
period in hours or in days is taken, and a number of counts in the file (constant, simply to indicate the "detail in which we
want to see the data), which is also retrieved from the
config file. This sets actual points in the curve of that plane, creating a first time series. This is the first step. For
this code it is
enough. It should return two plots: one with the curve in the flux vs eff plane, and one with the time series of flux and eff.
The time series should be saved in a csv file, with columns for time, flux, and eff. The plots should be saved as png files.

The idea of this script is basically generating a time series of a "constant flux but varying eff",
or a "Forbush Decrease" type of curve, or a "Ground Level Enhancement" type of curve, or
"constant eff but varying flux" type of curve. The idea is to be able to generate different types of
curves in the flux vs eff plane, and then use them as synthetic datasets for the next steps.
'''
