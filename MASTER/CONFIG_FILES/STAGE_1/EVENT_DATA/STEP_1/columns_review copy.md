


# Rename the variables

# Tasks 0, 1 and 2
Q1_B_1 --> p1_s1_eb_q
Q1_F_1 --> p1_s1_ef_q
T2_B_4 --> p2_s4_eb_t

# Tasks 2 and 3
Q1_Q_dif_4 --> p1_s4_qdif
Q1_Q_sum_1 --> p1_s1_qsum

# Tasks 3 and 4
P1_Q_dif_final --> p1_qdif
P1_Q_sum_final --> p1_qsum
P1_T_dif_final --> p1_tdif
P1_T_sum_final --> p1_tsum
P1_Y_final --> p1_ypos

# Task 4 and 5

ext_res_tdif_{plane} --> p{plane}_tdif_res_ext
res_tdif_{plane} --> p{plane}_tdif_res

ext_res_tsum_{plane} --> p{plane}_tsum_res_ext
res_tsum_{plane} --> p{plane}_tsum_res

ext_res_ystr_{plane} --> p{plane}_ystr_res_ext
res_ystr_{plane} --> p{plane}_ystr_res

ext_res_tdif_{plane}_err --> p{plane}_tdif_res_ext_err
res_tdif_{plane}_err --> p{plane}_tdif_res_err

ext_res_tsum_{plane}_err --> p{plane}_tsum_res_ext_err
res_tsum_{plane}_err --> p{plane}_tsum_res_err

ext_res_ystr_{plane}_err --> p{plane}_ystr_res_ext_err
res_ystr_{plane}_err --> p{plane}_ystr_res_err

x --> event_x
y --> event_y
xp --> event_xp
yp --> event_yp
s --> event_s
t0 --> event_t0
theta --> event_theta
phi --> event_phi
det_{variable} --> event_det_{variable}
tim_{variable} --> event_tim_{variable}
Time --> time

Q_P{plane}s{strip} --> [dont use them, since we have the already defined p{plane}_s{strip}_qsum]

charge_1 --> [dissapear and be replaced by p1_qsum, which already exists]
charge_2 --> [dissapear and be replaced by p2_qsum, which already exists]
charge_3 --> [dissapear and be replaced by p3_qsum, which already exists]
charge_4 --> [dissapear and be replaced by p4_qsum, which already exists]
charge_event --> event_charge

conv_distance --> timtrack_conv_distance
converged --> timtrack_converged
iterations --> timtrack_iterations

# All along the code
acq_tt   --> tt_task0_acq
raw_tt   --> tt_task0_raw
clean_tt --> tt_task1_clean
cal_tt   --> tt_task2_cal
list_tt  --> tt_task3_list
fit_tt   --> tt_task4_fit
post_tt  --> tt_task5_post

task1_problematic_channel_count --> filter_task1_problematic_channel_count
task1_problematic_channel_resolution_exact --> filter_task1_problematic_channel_exact
task2_problematic_strip_count --> filter_task2_problematic_strip_count
task2_problematic_strip_resolution_exact --> filter_task2_problematic_strip_exact
task3_problematic_plane_count --> filter_task3_problematic_plane_count
task3_problematic_plane_resolution_exact --> filter_task3_problematic_plane_exact
total_problematic_offender_count --> filter_total_problematic_offender_count

(CREATE:)            transferred_acq_to_raw
raw_to_clean_tt  --> transferred_raw_to_clean
clean_to_cal_tt  --> transferred_clean_to_cal
cal_to_list_tt   --> transferred_cal_to_list
fit_to_post_tt   --> transferred_fit_to_post



extension_tt --> [stop using, not needed]
processed_tt --> [stop using, not needed if we already have fit_tt and post_tt in tasks 4 and 5]
tracking_tt --> [stop using, not needed if we already have fit_tt and post_tt in tasks 4 and 5]

z_P{plane} --> z_p{plane}


-------------------------------------------------------------------




active_strips_P1
active_strips_P2
active_strips_P3
active_strips_P4

adj_dis



det_chi2_tsum_fit



delta_s




event_id






Phi_pred
plane_charge_topology_code


post_tt
processed_tt


region








th_chi_0
th_chi_3
th_chi_6



Theta_pred

tim_th_chi_sigmafit_1234



---





acq_tt
acquisition_type
datetime
event_id
param_hash
Q1_B_1
Q1_B_2
Q1_B_3
Q1_B_4
Q1_F_1
Q1_F_2
Q1_F_3
Q1_F_4
Q2_B_1
Q2_B_2
Q2_B_3
Q2_B_4
Q2_F_1
Q2_F_2
Q2_F_3
Q2_F_4
Q3_B_1
Q3_B_2
Q3_B_3
Q3_B_4
Q3_F_1
Q3_F_2
Q3_F_3
Q3_F_4
Q4_B_1
Q4_B_2
Q4_B_3
Q4_B_4
Q4_F_1
Q4_F_2
Q4_F_3
Q4_F_4
raw_tt
T1_B_1
T1_B_2
T1_B_3
T1_B_4
T1_F_1
T1_F_2
T1_F_3
T1_F_4
T2_B_1
T2_B_2
T2_B_3
T2_B_4
T2_F_1
T2_F_2
T2_F_3
T2_F_4
T3_B_1
T3_B_2
T3_B_3
T3_B_4
T3_F_1
T3_F_2
T3_F_3
T3_F_4
T4_B_1
T4_B_2
T4_B_3
T4_B_4
T4_F_1
T4_F_2
T4_F_3
T4_F_4
clean_tt
raw_to_clean_tt
task1_problematic_channel_count
task1_problematic_channel_resolution_exact
cal_tt
clean_to_cal_tt
P1_Q_total_uncal
P2_Q_total_uncal
P3_Q_total_uncal
P4_Q_total_uncal
Q1_Q_dif_1
Q1_Q_dif_2
Q1_Q_dif_3
Q1_Q_dif_4
Q1_Q_sum_1
Q1_Q_sum_2
Q1_Q_sum_3
Q1_Q_sum_4
Q2_Q_dif_1
Q2_Q_dif_2
Q2_Q_dif_3
Q2_Q_dif_4
Q2_Q_sum_1
Q2_Q_sum_2
Q2_Q_sum_3
Q2_Q_sum_4
Q3_Q_dif_1
Q3_Q_dif_2
Q3_Q_dif_3
Q3_Q_dif_4
Q3_Q_sum_1
Q3_Q_sum_2
Q3_Q_sum_3
Q3_Q_sum_4
Q4_Q_dif_1
Q4_Q_dif_2
Q4_Q_dif_3
Q4_Q_dif_4
Q4_Q_sum_1
Q4_Q_sum_2
Q4_Q_sum_3
Q4_Q_sum_4
T1_T_dif_1
T1_T_dif_2
T1_T_dif_3
T1_T_dif_4
T1_T_sum_1
T1_T_sum_2
T1_T_sum_3
T1_T_sum_4
T2_T_dif_1
T2_T_dif_2
T2_T_dif_3
T2_T_dif_4
T2_T_sum_1
T2_T_sum_2
T2_T_sum_3
T2_T_sum_4
T3_T_dif_1
T3_T_dif_2
T3_T_dif_3
T3_T_dif_4
T3_T_sum_1
T3_T_sum_2
T3_T_sum_3
T3_T_sum_4
T4_T_dif_1
T4_T_dif_2
T4_T_dif_3
T4_T_dif_4
T4_T_sum_1
T4_T_sum_2
T4_T_sum_3
T4_T_sum_4
task2_problematic_strip_count
task2_problematic_strip_resolution_exact
total_problematic_offender_count
active_strips_P1
active_strips_P2
active_strips_P3
active_strips_P4
adj_dis
cal_to_list_tt
list_tt
P1_Q_dif_final
P1_Q_sum_final
P1_T_dif_final
P1_T_sum_final
P1_Y_final
P2_Q_dif_final
P2_Q_sum_final
P2_T_dif_final
P2_T_sum_final
P2_Y_final
P3_Q_dif_final
P3_Q_sum_final
P3_T_dif_final
P3_T_sum_final
P3_Y_final
P4_Q_dif_final
P4_Q_sum_final
P4_T_dif_final
P4_T_sum_final
P4_Y_final
plane_charge_topology_code
task3_problematic_plane_count
task3_problematic_plane_resolution_exact
z_P1
z_P2
z_P3
z_P4
charge_1
charge_2
charge_3
charge_4
charge_event
det_chi2_tsum_fit
conv_distance
converged
delta_s
det_chi2
det_ext_res_tdif_1
det_ext_res_tdif_2
det_ext_res_tdif_3
det_ext_res_tdif_4
det_ext_res_tsum_1
det_ext_res_tsum_2
det_ext_res_tsum_3
det_ext_res_tsum_4
det_ext_res_ystr_1
det_ext_res_ystr_2
det_ext_res_ystr_3
det_ext_res_ystr_4
det_phi
det_processed_tt
det_res_tdif_1
det_res_tdif_2
det_res_tdif_3
det_res_tdif_4
det_res_tsum_1
det_res_tsum_2
det_res_tsum_3
det_res_tsum_4
det_res_ystr_1
det_res_ystr_2
det_res_ystr_3
det_res_ystr_4
det_s
det_s_ordinate
det_t0
det_th_chi
det_theta
det_x
det_y
ext_res_tdif_1
ext_res_tdif_1_err
ext_res_tdif_2
ext_res_tdif_2_err
ext_res_tdif_3
ext_res_tdif_3_err
ext_res_tdif_4
ext_res_tdif_4_err
ext_res_tsum_1
ext_res_tsum_1_err
ext_res_tsum_2
ext_res_tsum_2_err
ext_res_tsum_3
ext_res_tsum_3_err
ext_res_tsum_4
ext_res_tsum_4_err
ext_res_ystr_1
ext_res_ystr_1_err
ext_res_ystr_2
ext_res_ystr_2_err
ext_res_ystr_3
ext_res_ystr_3_err
ext_res_ystr_4
ext_res_ystr_4_err
extension_tt
fit_tt
iterations
processed_tt
Q_P1s1
Q_P1s2
Q_P1s3
Q_P1s4
Q_P2s1
Q_P2s2
Q_P2s3
Q_P2s4
Q_P3s1
Q_P3s2
Q_P3s3
Q_P3s4
Q_P4s1
Q_P4s2
Q_P4s3
Q_P4s4
res_tdif_1
res_tdif_1_err
res_tdif_2
res_tdif_2_err
res_tdif_3
res_tdif_3_err
res_tdif_4
res_tdif_4_err
res_tsum_1
res_tsum_1_err
res_tsum_2
res_tsum_2_err
res_tsum_3
res_tsum_3_err
res_tsum_4
res_tsum_4_err
res_ystr_1
res_ystr_1_err
res_ystr_2
res_ystr_2_err
res_ystr_3
res_ystr_3_err
res_ystr_4
res_ystr_4_err
s
s_err
t0
t0_err
th_chi_0
th_chi_3
th_chi_6
theta
theta_err
tim_charge_1
tim_charge_2
tim_charge_3
tim_charge_4
tim_charge_event
tim_conv_distance
tim_converged
tim_ext_res_tdif_1
tim_ext_res_tdif_2
tim_ext_res_tdif_3
tim_ext_res_tdif_4
tim_ext_res_tsum_1
tim_ext_res_tsum_2
tim_ext_res_tsum_3
tim_ext_res_tsum_4
tim_ext_res_ystr_1
tim_ext_res_ystr_2
tim_ext_res_ystr_3
tim_ext_res_ystr_4
tim_iterations
tim_list_tt
tim_phi
tim_res_td
tim_res_tdif_1
tim_res_tdif_2
tim_res_tdif_3
tim_res_tdif_4
tim_res_ts
tim_res_tsum_1
tim_res_tsum_2
tim_res_tsum_3
tim_res_tsum_4
tim_res_y
tim_res_ystr_1
tim_res_ystr_2
tim_res_ystr_3
tim_res_ystr_4
tim_s
tim_t0
tim_th_chi
tim_th_chi_sigmafit_1234
tim_theta
tim_x
tim_xp
tim_y
tim_yp
tracking_tt
x
x_err
xp
y
y_err
yp
fit_to_post_tt
Phi_pred
post_tt
region
Theta_pred
Time