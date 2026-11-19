# Implementer Trade-Off Guidelines

* **Alignment Window vs. Margin Peak:** Shortening the margin alignment window boosts cycle speed but raises transient handoff demand. Pairing a smooth alignment profile with a post-margin focal guard band (≤ 0.25 nm) keeps margin spikes controlled even under 2.5x disturbance plus initial offset.
* **Etch/Dissolution Rate vs. Cycle Time:** Higher effective etch/dissolution rates and shorter settle holds reduce cycle time past 50% gain provided focal error stays inside tolerance during active etching.
* **Wafer Swap & Maintenance Schedules:** Implementers select wafer swap, maintenance, and calibration intervals suited to their load-lock, handling, and metrology workflow; swap windows also provide passive thermal relaxation between wafers.
* **Thermal Baseline Scaling:** The pre-diamond thermal-drain baseline satisfies current power loads and cycle counts; higher-dissipation cooling is only needed if future deployments increase continuous power/duty cycles past thermal-drift limits.
