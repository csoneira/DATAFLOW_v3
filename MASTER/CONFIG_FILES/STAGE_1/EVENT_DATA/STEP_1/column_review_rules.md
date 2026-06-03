

# Tasks 0, 1 and 2
Q{plane}_{F/B}_{strip} --> p{plane}_s{strip}_e{f/b}_q
T{plane}_{F/B}_{strip} --> p{plane}_s{strip}_e{f/b}_t
column_6 --> acquisition_type, but it alreyy should be like that

# Tasks 2 and 3
Q{plane}_Q_dif_{strip} --> p{plane}_s{strip}_qdif
Q{plane}_Q_sum_{strip} --> p{plane}_s{strip}_qsum
T{plane}_T_dif_{strip} --> p{plane}_s{strip}_tdif
T{plane}_T_sum_{strip} --> p{plane}_s{strip}_tsum

P{plane}_Q_total_uncal --> p{plane}_qsum_uncalibrated


# Tasks 3 and 4
P{plane}_Q_dif_final --> p{plane}_qdif
P{plane}_Q_sum_final --> p{plane}_qsum
P{plane}_T_dif_final --> p{plane}_tdif
P{plane}_T_sum_final --> p{plane}_tsum
P{plane}_Y_final --> p{plane}_ypos


adj_dis --> [TOTALLY ERASE WITH ALL THE REFERENCES TO IT]

th_chi_{ndf} --> th_chisq_df_{ndf}

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

Time --> datetime (so actually no need to change the name of variable to Time, because we alreday start from datetime from the begining and we will carry it till the end)

x_err     --> event_x_err
y_err     --> event_y_err
s_err     --> event_s_err
t0_err    --> event_t0_err
theta_err --> event_theta_err
phi_err   --> event_phi_err

delta_s --> event_s_err [NOTE THAT ITS THE SAME, so actually you dont need to define delta_s especifically, but use the s_err]

Q_P{plane}s{strip} --> [dont use them, since we have the already defined p{plane}_s{strip}_qsum]

charge_1 --> [dissapear and be replaced by p1_qsum, which already exists]
charge_2 --> [dissapear and be replaced by p2_qsum, which already exists]
charge_3 --> [dissapear and be replaced by p3_qsum, which already exists]
charge_4 --> [dissapear and be replaced by p4_qsum, which already exists]
charge_event --> event_charge

conv_distance --> timtrack_conv_distance
converged --> timtrack_converged
iterations --> timtrack_iterations



# Task 5
Phi_pred --> REMOVE TOTALLY AND AL THE CODE RELATED TO IT in task 5
Theta_pred --> REMOVE TOTALLY AND ALL THE CODE RELATED TO IT in task 5
region --> REMOVE TOTALLY AND ALL THE CODE RELATED TO IT in task 5




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

topology_task1_channel --> [NEW: IT IS A 32 BIT VALUE (BECAUSE THERE ARE 32 CHANNELS) THAT WILL ENCODE IF THE CHANNEL WAS 0 OR NOT 0, BASED ON CHARGE]
plane_charge_topology_code --> topology_task2_strip (but define it already in task 2, dont wait until task 3. IT IS A 16 BIT VALUE (BECAUSE THERE ARE 16 STRIPS) THAT WILL ENCODE IF THE STRIP WAS 0 OR NOT 0, BASED ON CHARGE)
topology_task3_plane --> [NEW: IT IS A 4 BIT VALUE (BECAUSE THERE ARE 4 PLANES) THAT WILL ENCODE IF THE PLANE WAS 0 OR NOT 0, BASED ON CHARGE]


active_strips_P{plane} --> TRY TO NOT USE IT, SINCE WE WILL ALREADY HAVE DEFINED topology_task2_strip IN A PREVIOUS STEP


*_crstlk --> [remove totally all code that makes reference to these variables, we are not interested in them anymore]

---


# Keep untouched

acquisition_type
event_id
param_hash


---

