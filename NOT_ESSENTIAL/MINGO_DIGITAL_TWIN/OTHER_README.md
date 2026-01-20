flowchart TB

%% ============================================================
%% MINGO_DIGITAL_TWIN: Domain-separated scheme with in-line boundaries
%% Requested domains:
%%   - MUONS
%%   - RPC: (Active Volume / Avalanche) and (Induction on readout)
%%   - ELECTRONICS: (Connectors) (FEE) (TDC)
%% ============================================================

subgraph MU["MUONS (primary particle + transport)"]
  direction TB
  S1["STEP 1: Muon generation\n- Sample (x0,y0,z0), (theta,phi), v≈c, T0\n- Standard flux/angle distributions\nOutput: muon_sample_<N>"]:::mu
  S2["STEP 2: Muon propagation and plane crossings\n- Straight-line transport through station geometry\n- For each plane i: (xi,yi,zi), ti = T0 + Li/v\nOutput: geom_<G>"]:::mu
end

B1["BOUNDARY: detector response starts\nRPC Active Volume (gas gap)"]:::boundary

subgraph RPC["RPC (detector response)"]
  direction TB

  subgraph RPC_AV["RPC Active Volume (gas gap microphysics)"]
    direction TB
    S3["STEP 3: Ionization + avalanche in gas gap (NO readout)\n- N_ion, avalanche size Ne\n- Avalanche centroid/time (xa,ya,ta)\nOutput: geom_<G>_avalanche"]:::rpc_av
  end

  B2["BOUNDARY: readout coupling starts\nRPC Induction on Cu strips"]:::boundary

  subgraph RPC_IND["RPC Induction + Readout Plane"]
    direction TB
    S4["STEP 4: Induction footprint and affected strips\n- Map (xa,ya,Ne) to induced footprint\n- Determine hit strips and per-strip induced quantities\nOutput: geom_<G>_hit"]:::rpc_ind
    S5["STEP 5: Strip observables (still on-detector)\n- q_diff (charge imbalance/uncertainty proxy)\n- T_diff from x-position along strip\nOutput: geom_<G>_signal"]:::rpc_ind
    S6["STEP 6: Propagation along strip to both ends\n- Compute T_front/T_back and Q_front/Q_back\nOutput: geom_<G>_frontback"]:::rpc_ind
  end
end

B3["BOUNDARY: electronics chain starts\nConnectors/cables to FEE inputs"]:::boundary

subgraph ELEC["ELECTRONICS + DAQ"]
  direction TB

  subgraph CONN["Connectors / Cables (passive delays)"]
    direction TB
    S7["STEP 7: Connector/cable travel-time offsets\n- Apply Δt offsets from strip ends to FEE inputs\nOutput: geom_<G>_calibrated"]:::conn
  end

  B4["BOUNDARY: active electronics response starts\nFEE threshold + time pickoff"]:::boundary

  subgraph FEE["FEE (front-end electronics)"]
    direction TB
    S8["STEP 8: FEE model\n- Apply charge threshold Q_thr\n- Charge-to-time conversion / time-walk\nOutput: geom_<G>_threshold"]:::fee
  end

  B5["BOUNDARY: digitization and event definition\nTrigger logic + TDC time model"]:::boundary

  subgraph TDC["Trigger + TDC (DAQ logic and digitization)"]
    direction TB
    S9["STEP 9: Trigger coincidence selection\n- Keep only allowed plane combinations\nOutput: geom_<G>_triggered"]:::tdc
    S10["STEP 10: TDC/DAQ timing model\n- Apply TDC smear and clock jitter\nOutput: geom_<G>_daq"]:::tdc
  end
end

B6["BOUNDARY: formatting only\nEvent builder emulation"]:::boundary

subgraph OUT["OUTPUT (representation only)"]
  direction TB
  SF["STEP FINAL: Event builder formatting\n- Serialize to station-style .dat\nOutput: SIMULATED_DATA/mi0XYYDDDHHMMSS.dat"]:::fmt
end

%% -------------------------
%% Causal order (single main line)
%% -------------------------
S1 --> S2 --> B1 --> S3 --> B2 --> S4 --> S5 --> S6 --> B3 --> S7 --> B4 --> S8 --> B5 --> S9 --> S10 --> B6 --> SF

%% -------------------------
%% Styles (domain-coded)
%% -------------------------
classDef mu fill:#eef7ff,stroke:#1b4965,stroke-width:1px,color:#0b2233;
classDef rpc_av fill:#eafaf1,stroke:#2d6a4f,stroke-width:1px,color:#0b2233;
classDef rpc_ind fill:#fff7e6,stroke:#8a5a00,stroke-width:1px,color:#0b2233;
classDef conn fill:#f7f0ff,stroke:#5a189a,stroke-width:1px,color:#0b2233;
classDef fee  fill:#f0e6ff,stroke:#5a189a,stroke-width:1px,color:#0b2233;
classDef tdc  fill:#fff0f3,stroke:#9d0208,stroke-width:1px,color:#0b2233;
classDef fmt  fill:#f2f2f2,stroke:#333333,stroke-width:1px,color:#0b2233;
classDef boundary fill:#ffffff,stroke:#555555,stroke-width:1px,color:#222222,stroke-dasharray: 4 3;
