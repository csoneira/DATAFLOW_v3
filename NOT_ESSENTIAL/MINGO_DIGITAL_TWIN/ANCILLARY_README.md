Below is a compact, detection-physics-aligned scheme that enforces clear boundaries between (A) particle generation and transport, (B) detector microphysics, (C) induction and strip readout physics, (D) electronics and DAQ logic, and (E) output formatting.

---

## End-to-end causal chain (what causes what)

### Layered flow (physics to data)

1. **Primary particle state** (muon phase space)
2. **Geometrical transport** (muon intersections with planes)
3. **RPC microphysics** (ionization and avalanche in gas gap)
4. **Electromagnetic induction on readout** (strip coupling, affected strips)
5. **In-strip observable construction** (time-difference and charge-difference proxies)
6. **Signal propagation along strip** (front/back times and charges)
7. **Passive delays to electronics** (connectors/cables to FEE)
8. **Front-end electronics response** (thresholding and time pickoff)
9. **Trigger/selection logic** (coincidence patterns)
10. **TDC digitization effects** (jitter, smear, clock granularity)
11. **Event builder formatting** (station-style .dat)

---

## Scheme as a step-by-step pipeline (aligned to your simulator)

### STEP 1: Muon generation (Blank → Generated)

**Domain:** Primary particle model (no detector).
**Physical content:** Sample muon initial conditions from standard flux and angular distributions.

* **State defined:** [(x_0,y_0,z_0),\ \hat{u}(\theta,\phi),\ v\approx c,\ T_0]
* **Outputs (artifact):** `muon_sample_<N>`
* **Key fields:** `X_gen, Y_gen, Z_gen, Theta_gen, Phi_gen, T0_ns`

**Causality to next step:** Defines the initial 4D trajectory used for geometric intersections.

---

### STEP 2: Muon propagation through station geometry (Generated → Crossing)

**Domain:** Deterministic transport in geometry (still no detector response).
**Physical content:** Extend the straight trajectory and compute intersection points and times at each plane.

* **Core operation:** For each plane (i), solve intersection and compute:

  * **Incidence position:** ((x_i,y_i,z_i))
  * **Incidence time:** (t_i = T_0 + L_i / v) (with (L_i) path length to plane)
* **Outputs:** `geom_<G>`
* **Key fields:** per-plane positions `X_gen_i, Y_gen_i, Z_gen_i` and timing `T_sum_i_ns`, plus `tt_crossing`

**Boundary note:** This step is pure geometry and kinematics; no gas, no charge, no readout.

---

### STEP 3: RPC ionization and avalanche in gas gap (Crossing → Avalanche)

**Domain:** Detector active volume microphysics (no readout).
**Physical content:** Given a muon crossing the active gap, generate ionization statistics and avalanche properties.

* **Core stochastic elements:**

  * Number of primary ionizations (N_{\mathrm{ion}})
  * Avalanche gain or total electrons (N_e)
  * Avalanche centroid position ((x_a,y_a)) within gap and time (t_a)
* **Outputs:** `geom_<G>_avalanche`
* **Key fields:** `avalanche_size_electrons_i, avalanche_x_i, avalanche_y_i, tt_avalanche`

**Critical separation:** This step stops at the gas response. It must not depend on strip pitch, thresholds, cable delays, or electronics.

---

### STEP 4: Induction and strip coupling (Avalanche → Hit)

**Domain:** Readout physics begins here (induction on Cu strips).
**Physical content:** Use avalanche centers and electron counts to estimate the induced footprint on the pickup plane and decide which strips are affected.

* **Core operation:** Map ((x_a,y_a,N_e)) → induced charge distribution across strips:

  * Identify strip indices (s_j) affected by the induction area
  * Compute per-strip induced quantities (e.g., charge sharing, centroid)
* **Outputs:** `geom_<G>_hit`
* **Key fields:** `Y_mea_i_sj (qsum), X_mea_i_sj, T_sum_meas_i_sj, tt_hit`

**Boundary note:** This is still “detector + readout plane physics” and not yet “electronics”.

---

### STEP 5: Complete strip-level observables (Hit → Signal)

**Domain:** Strip-internal readout model (still on-detector).
**Physical content:** Convert avalanche/induction information into two derived observables:

1. **Charge imbalance proxy:** (q_{\mathrm{diff}}) (charge uncertainty / asymmetry model)
2. **Time-difference proxy:** (T_{\mathrm{diff}}) from the avalanche (x)-position along the strip

* **Conceptual mapping:**

  * (q_{\mathrm{diff}} = f(Q_{\mathrm{ind}}, \sigma_Q, \text{sharing model}))
  * (T_{\mathrm{diff}} \propto (2x/L)\cdot (L/v_{\mathrm{prop}})) (sign encodes nearer end)
* **Outputs:** `geom_<G>_signal`
* **Key fields:** `T_diff_i_sj, q_diff_i_sj`

**Separation rationale:** This step defines “what the strip *would* report at its ends,” but does not yet propagate to ends.

---

### STEP 6: Propagation to strip ends (Signal → Front/Back)

**Domain:** Transmission-line propagation on the strip.
**Physical content:** Convert ((T_{\mathrm{diff}}, q_{\mathrm{diff}})) into end-specific times and charges.

* **Core operation:** For each strip hit:

  * (T_{\mathrm{front}}, T_{\mathrm{back}}) from propagation speed and hit position
  * (Q_{\mathrm{front}}, Q_{\mathrm{back}}) from charge sharing / attenuation model
* **Outputs:** `geom_<G>_frontback`
* **Key fields:** `T_front_i_sj, T_back_i_sj, Q_front_i_sj, Q_back_i_sj`

---

### STEP 7: Connector/cable delays to FEE (Front/Back → Calibrated)

**Domain:** Passive routing and calibration offsets.
**Physical content:** Add travel-time offsets (and optionally attenuation) from strip ends through connectors to the front-end.

* **Core operation:**

  * (T'*{\mathrm{front}} = T*{\mathrm{front}} + \Delta t_{\mathrm{cable,front}})
  * (T'*{\mathrm{back}} = T*{\mathrm{back}} + \Delta t_{\mathrm{cable,back}})
* **Outputs:** `geom_<G>_calibrated`
* **Key fields:** calibrated front/back times and charges

**Boundary note:** Still analog-domain timing; digitization has not occurred.

---

### STEP 8: Front-end electronics (Threshold / FEE)

**Domain:** Electronics response and discrimination.
**Physical content:** Apply charge threshold and convert charge to timing (time walk / charge-to-time model).

* **Core operation:**

  * Discard signals with (Q < Q_{\mathrm{thr}})
  * Compute time pickoff (t_{\mathrm{FEE}} = g(Q)) (e.g., leading-edge walk model)
* **Outputs:** `geom_<G>_threshold`
* **Key fields:** thresholded charges and converted times

**Critical separation:** This is the first step that depends on FEE parameters.

---

### STEP 9: Trigger logic and coincidence selection (Threshold → Trigger)

**Domain:** DAQ logic (event definition begins here).
**Physical content:** Keep only plane combinations that satisfy the configured trigger.

* **Core operation:** Apply coincidence pattern mask over planes:

  * Accept event if planes (\in) allowed set and within coincidence window
* **Outputs:** `geom_<G>_triggered`
* **Key fields:** `tt_trigger`

---

### STEP 10: TDC and DAQ clock effects (Triggered → Jitter)

**Domain:** Digitization artifacts.
**Physical content:** Apply TDC smear and clock jitter to the accepted coincidence events.

* **Core operation:**

  * (t_{\mathrm{dig}} = t_{\mathrm{FEE}} + \delta t_{\mathrm{TDC}} + \delta t_{\mathrm{clk}})
* **Outputs:** `geom_<G>_daq`
* **Key fields:** `daq_jitter_ns`, jittered `T_front/T_back`

---

### STEP FINAL: Event builder and station-style formatting (DAQ → Station .dat)

**Domain:** Output emulation (format-level realism).
**Physical content:** Transform digitized hits into the same structural format as real data output.

* **Core operation:** Serialize per-strip timing/charge records into station `.dat` layout and naming convention.
* **Outputs:** `SIMULATED_DATA/mi0XYYDDDHHMMSS.dat` plus registry

**Separation note:** This step reshapes representation; it should not alter physics.

---

## Explicit boundary markers (recommended to document prominently)

1. **Start of detector response:** **STEP 3** (gas ionization + avalanche)
2. **Start of readout coupling:** **STEP 4** (induction and strip selection)
3. **Start of electronics:** **STEP 8** (threshold and time pickoff)
4. **Start of DAQ-defined “events”:** **STEP 9** (trigger coincidence acceptance)
5. **Start of digitization artifacts:** **STEP 10** (TDC jitter/smear)
6. **Start of formatting-only operations:** **STEP FINAL**

---

## Minimal “scheme diagram” (copy/paste into a README)

**Physics chain:**
Muon generation (1) → Geometric crossings (2) → Gas avalanche (3) → Induction/strips (4) → Strip observables (T_{\mathrm{diff}},q_{\mathrm{diff}}) (5)

**Readout and electronics chain:**
Front/back propagation (6) → Cable/connector delays (7) → FEE threshold + time pickoff (8)

**DAQ and output chain:**
Trigger coincidence (9) → TDC jitter/smear (10) → Event builder formatting (FINAL)

---

If you want, I can convert the above into a single-page figure (for documentation) using a block diagram notation (for example, Mermaid flowchart text or a LaTeX TikZ diagram), preserving the same boundary markers and step numbering.
