# Precision Split-Z Multi-Path Lithography Etching Subsystem

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=white)
![Simulation: NumPy + Matplotlib](https://img.shields.io/badge/Simulation-NumPy%20%2B%20Matplotlib-013243.svg)
![AI Collaborator: Google Gemini](https://img.shields.io/badge/AI%20Collaborator-Google%20Gemini-4285F4.svg)
![Multi-Physics & Control Support: Team Gemini](https://img.shields.io/badge/Multi--Physics%20%26%20Control%20Support-Team%20Gemini-34A853.svg)

- **Architecture Overview:** Hardware-agnostic, multi-physics lithographic etching control model pairing coarse substrate datum alignment (Z<sub>sub</sub>) with fast per-path focus and attitude trimming (Z<sub>lens</sub>, θ<sub>x</sub>, θ<sub>y</sub>) across four disjoint active quadrants on a circular substrate.
- **Control & Scheduling:** Combines rate-aware hysteresis pre-compensation, boundary-layer sliding-mode control (10 kHz discrete step, Δt = 100 μs), alternating deep/shallow site ordering, and an outer margin alignment window so initial datum handoff transients stay outside active production etching.
- **Thermal Baseline:** Evaluates a pre-diamond thermal-drain baseline across single-wafer, 2.0 s, and multi-wafer campaigns with wafer-swap thermal relaxation windows, tracking temperature rise (ΔT) and heat generation vs. external dissipation before adding higher-complexity heat sinking.
- **Key Verified Benchmarks (v11–v12):**
  - **Production Focus Accuracy:** ~36.6–38.1 pm RMS focal error and worst-case production peak ≤ 0.1134 nm under 2.5x disturbance plus initial offset (well within a 0.30 nm sub-nm focus budget and ±22.31 nm High-NA EUV Rayleigh half-DOF at λ = 13.5 nm, NA = 0.55).
  - **Margin Transient Containment:** Worst-case margin alignment peak ≤ 0.1839 nm with unclipped actuator force authority (saturation ratio ~0.111–0.112 &lt; 1.0).
  - **Throughput & Coverage:** Over 50% cycle time reduction vs. v9 baseline (78.40 ms mean active etch+dev and 153.40 ms total cycle with 75.0 ms wafer swap in v11) with 100% target site completion (320/320 in v11; 800/800 across v12) and zero missed or overlapped active cells.

## Core Challenges & Mitigation Solutions

- **Sub-Nanometer Focus & Vibration/Disturbance Sensitivity:**
  - *Challenge:* High-NA EUV lithography (λ = 13.5 nm, NA = 0.55) operates within a tight Rayleigh depth-of-focus budget (±22.31 nm half-DOF; ≤ 0.30 nm sub-nm focus allocation) while floor vibrations, acoustic/structural disturbances, and stage motion excite focus and attitude errors. Reference context: [MIT Fab Lab Wafer Scanner Control Overview](https://fab.cba.mit.edu/classes/865.24/topics/mechanical_design/mechanical_design/precision-frontier/Control_of_Wafer_Scanners_Methods_and_Developments.pdf) and [Berkeley MSC Wafer Scanning Research](https://msc.berkeley.edu/research/wafer.html).
  - *Mitigation:* Splits actuation between a coarse/infrequent substrate datum channel (Z<sub>sub</sub>) and four per-branch fast focus/tilt channels (Z<sub>lens</sub>, θ<sub>x</sub>, θ<sub>y</sub>) running at 10 kHz (Δt = 100 μs) with a first-order handoff filter, holding ~36.6–38.1 pm RMS focal error and ≤ 0.1134 nm worst-case production peak under 2.5x disturbance plus initial offset (`v11`–`v12`).

- **Actuator Hysteresis & Force Clipping / Saturation Transients:**
  - *Challenge:* Precision actuators and flexures exhibit rate-dependent hysteresis and peak force limits; excessive raw control gains or abrupt stiffness feedforward can drive force saturation (seen in `v10` at 25.355x), causing slew lag, overshoot, and extra thermal load. Reference context: [TU/e Multi-Body Lithography Stage Study](https://research.tue.nl/files/196152956/1_s2.0_S0094114X21003712_main.pdf) and [arXiv Global Iterative Sliding Mode Control](https://arxiv.org/html/2103.12580v2).
  - *Mitigation:* Combines rate-aware hysteresis pre-compensation with boundary-layer sliding-mode control and balanced force authority (`v11`–`v12`), bringing peak force saturation ratio down to ~0.111–0.112 (&lt; 1.0, unclipped).

- **Initial Alignment Handoff Transients & Spatial Coverage Gaps/Overlaps:**
  - *Challenge:* Initial coarse datum shifts and settling transients can affect early etching if active processing begins prematurely, while uncoordinated multi-path partitioning risks missed or overlapped zones on a circular wafer. Reference context: [University of Michigan Ultra-Precision Step Stage Report](https://scholarships.engin.umich.edu/wp-content/uploads/sites/36/2018/06/2017-Wang-Chu-Chien-Wen-Report-Bowen-Zeng.pdf).
  - *Mitigation:* Uses a circular substrate layout with an outer margin alignment window (1.65 &lt; r ≤ 2.00) and a post-margin focal guard band (≤ 0.25 nm) prior to etching, paired with four disjoint active quadrants inside R<sub>active</sub> = 1.65 and alternating deep/shallow site ordering, achieving ≤ 0.1839 nm worst-case margin peak and 100% site completion with zero missed or overlapped active cells.

- **Cycle Time Bottlenecks & Multi-Wafer Thermal Accumulation:**
  - *Challenge:* Slow settle-and-etch cycles constrain wafer throughput, while back-to-back runs can accumulate temperature rise (ΔT) and thermal drift if heat generation outpaces external dissipation. Reference context: [LBNL / EUV Litho High-NA Wavefront & Thermal Context](https://www.euvlitho.com/2019/P61.pdf).
  - *Mitigation:* Accelerates settle-gated etch/dissolution and schedules wafer-swap thermal relaxation intervals on a pre-diamond thermal-drain baseline, achieving &gt; 50% cycle time reduction vs. `v9` baseline with sub-mK to low-mK ΔT rise across multi-wafer runs.

## Repository Layout

- **src/:** Progression of simulation testbenches (`sim_testbench_v5.py` through `sim_testbench_v12_stress.py`).
- **docs/:** Comprehensive development report (`docs/comprehensive_development_report.md`), SI unit mapping (`docs/si_unit_mapping.md`), and operational trade-off guidelines (`docs/trade_off_guidelines.md`).
- **figures/:** Generated plot outputs (`*.png`) from simulation runs.
- **References.md:** Literature/case-study pointers and intellectual property notice.

```
split-z-4path-lithography-control/
├── README.md
├── LICENSE
├── References.md
├── requirements.txt
├── src/
│   ├── sim_testbench_v5.py
│   ├── sim_testbench_v6.py
│   ├── sim_testbench_v7.py
│   ├── sim_testbench_v8.py
│   ├── sim_testbench_v9.py
│   ├── sim_testbench_v10.py
│   ├── sim_testbench_v11.py
│   └── sim_testbench_v12_stress.py
├── docs/
│   ├── comprehensive_development_report.md
│   ├── si_unit_mapping.md
│   └── trade_off_guidelines.md
└── figures/
    ├── split_z_4path_v5.png
    ├── split_z_4path_v6.png
    ├── split_z_4path_v7.png
    ├── split_z_circular_v8.png
    ├── split_z_wafer_day_v9.png
    ├── split_z_physics_verify_v10.png
    ├── split_z_physics_verify_v11.png
    └── split_z_stress_v12.png
```
## Quick Start

```bash
pip install -r requirements.txt
python src/sim_testbench_v11.py
python src/sim_testbench_v12_stress.py
```
