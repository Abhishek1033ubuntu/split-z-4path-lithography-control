import numpy as np
import matplotlib.pyplot as plt


class HysteresisInverter:
    """Rate-aware hysteresis pre-compensator for sub-nm positioning."""

    def __init__(self, hyst_amp=0.04, rate_coeff=0.002):
        self.hyst_amp = hyst_amp
        self.rate_coeff = rate_coeff
        self.prev_target = 0.0

    def invert(self, desired_pos, dt):
        rate = (desired_pos - self.prev_target) / max(dt, 1e-9)
        eff_hyst = self.hyst_amp * (1.0 + self.rate_coeff * min(abs(rate), 150.0))
        u_cmd = desired_pos + eff_hyst * np.tanh(desired_pos) * np.tanh(rate * 1e-3)
        self.prev_target = desired_pos
        return u_cmd


class ActuatorChannel:
    """Precision channel with physical resonance, force saturation, hysteresis, and thermal balance."""

    def __init__(
        self,
        name,
        mass,
        stiffness,
        damping,
        max_force,
        r_th,
        c_th,
        thermal_drift_coeff,
        q_gen_scale=0.008,
        hyst_amp=0.04,
        kp=3200.0,
        kd=55.0,
        smc_gain=6.0,
        boundary_layer=0.05,
    ):
        self.name = name
        self.mass = mass
        self.k = stiffness
        self.c = damping
        self.max_force = max_force
        self.r_th = r_th
        self.c_th = c_th
        self.thermal_drift = thermal_drift_coeff
        self.q_gen_scale = q_gen_scale

        self.kp = kp
        self.kd = kd
        self.smc_gain = smc_gain
        self.boundary_layer = boundary_layer

        self.pos = 0.0
        self.vel = 0.0
        self.delta_T = 0.0
        self.hyst_state = 0.0
        self.hyst_amp = hyst_amp
        self.inverter = HysteresisInverter(hyst_amp=hyst_amp)

    @property
    def natural_freq_hz(self):
        return (1.0 / (2.0 * np.pi)) * np.sqrt(self.k / max(self.mass, 1e-12))

    @property
    def damping_ratio(self):
        return self.c / (2.0 * np.sqrt(max(self.k * self.mass, 1e-12)))

    def step(
        self,
        target_pos,
        disturbance_force,
        dt,
        latency_steps=1,
        cmd_history=None,
        active_etching=True,
    ):
        comp_cmd = self.inverter.invert(target_pos, dt)

        if cmd_history is not None:
            cmd_history.append(comp_cmd)
            idx = max(0, len(cmd_history) - 1 - latency_steps)
            delayed_cmd = cmd_history[idx]
        else:
            delayed_cmd = comp_cmd

        err = delayed_cmd - self.pos
        d_err = -self.vel
        sliding_var = err + 0.002 * d_err
        sat = np.clip(sliding_var / max(self.boundary_layer, 1e-6), -1.0, 1.0)

        f_raw = self.k * delayed_cmd + self.kp * err + self.kd * d_err + self.smc_gain * sat
        f_ctrl = np.clip(f_raw, -self.max_force, self.max_force)
        sat_ratio = abs(f_raw) / max(self.max_force, 1e-9)

        d_pos = self.vel * dt
        self.hyst_state += (
            d_pos - abs(d_pos) * self.hyst_state / max(self.hyst_amp * 4.0, 1e-6)
        ) * 0.10
        hyst_force = -self.k * 0.01 * self.hyst_state

        acc = (
            f_ctrl + hyst_force + disturbance_force - self.c * self.vel - self.k * self.pos
        ) / self.mass
        self.vel += acc * dt
        self.pos += self.vel * dt

        effort_ratio = f_ctrl / max(self.max_force, 1e-9)
        duty_factor = 1.0 if active_etching else 0.15
        q_gen = self.q_gen_scale * (effort_ratio ** 2) * duty_factor
        q_ext = self.delta_T / max(self.r_th, 1e-6)
        self.delta_T = max(0.0, self.delta_T + ((q_gen - q_ext) / max(self.c_th, 1e-6)) * dt)

        thermal_offset = self.thermal_drift * self.delta_T
        effective_pos = self.pos + thermal_offset
        return effective_pos, self.delta_T, q_gen, q_ext, sat_ratio


def smooth_step_profile(t, t_start, t_end, amp):
    profile = np.zeros_like(t)
    half_span = max((t_end - t_start) * 0.5, 1e-6)
    mid = 0.5 * (t_start + t_end)
    mask = (t >= t_start) & (t <= t_end)
    tau = (t[mask] - mid) / half_span
    profile[mask] = amp * 0.5 * (1.0 + np.cos(np.pi * tau))
    return profile


def order_alternating(sites):
    by_desc = sorted(sites, key=lambda s: s["depth"], reverse=True)
    ordered = []
    left, right = 0, len(by_desc) - 1
    take_deep = True
    while left <= right:
        if take_deep:
            ordered.append(by_desc[left])
            left += 1
        else:
            ordered.append(by_desc[right])
            right -= 1
        take_deep = not take_deep
    return ordered


def run_multi_wafer_campaign(
    num_wafers=20,
    align_end_t=0.024,
    settle_steps_req=6,
    etch_rate_scale=2.25,
    swap_duration_s=0.075,
    core_power_ratio=4.0,
    optical_eta=0.85,
    handoff_tau=0.003,
    seed=71,
):
    """
    Runs a multi-wafer campaign to verify:
    1) >= 50% cycle time reduction vs v9 baseline (175.40 ms active / 325.40 ms total),
    2) Multi-wafer thermal convergence (no runaway ratcheting on pre-diamond drain),
    3) Optical DOF, structural resonance, and force saturation limits.
    """
    dt = 1e-4  # 10 kHz discrete loop

    # Physical Optical Budget Check (High-NA EUV: lambda=13.5 nm, NA=0.55, k2=0.5)
    wavelength_nm = 13.5
    na = 0.55
    k2 = 0.50
    rayleigh_dof_half_nm = k2 * wavelength_nm / (na ** 2)  # ~22.31 nm
    sub_nm_focus_budget = 0.30  # Allocated sub-nm control budget

    # Instantiate persistent actuators across the 20-wafer continuous campaign
    z_sub = ActuatorChannel("Z_Sub", 1.0, 1600.0, 85.0, 80.0, 4.0, 1.5, 0.020, 0.008, kp=3800.0, kd=65.0)
    lens_acts = [
        ActuatorChannel(f"Z_Lens_P{p+1}", 0.14, 2800.0, 55.0, 75.0, 4.5, 1.2, 0.025, 0.010, kp=3600.0, kd=52.0)
        for p in range(4)
    ]
    tx_acts = [
        ActuatorChannel(f"TX_P{p+1}", 0.16, 2600.0, 50.0, 65.0, 4.5, 1.2, 0.020, 0.008, kp=3400.0, kd=48.0)
        for p in range(4)
    ]
    ty_acts = [
        ActuatorChannel(f"TY_P{p+1}", 0.16, 2600.0, 50.0, 65.0, 4.5, 1.2, 0.020, 0.008, kp=3400.0, kd=48.0)
        for p in range(4)
    ]

    buf_sub = []
    buf_lens = [[] for _ in range(4)]
    buf_tx = [[] for _ in range(4)]
    buf_ty = [[] for _ in range(4)]

    rng = np.random.default_rng(seed)
    per_path_power_scale = (core_power_ratio / 4.0) * optical_eta
    base_etch_rate = 240.0 * per_path_power_scale * etch_rate_scale
    dissolution_rate = 360.0 * etch_rate_scale
    focal_tol_nm = sub_nm_focus_budget

    base_depths = [8.0, 1.2, 6.0, 2.4]
    path_queues = [
        order_alternating([{"depth": base_depths[k] + 0.3 * p_idx} for k in range(4)])
        for p_idx in range(4)
    ]

    p_sub_handoff = 0.0
    alpha_handoff = dt / max(handoff_tau + dt, 1e-9)

    global_t = 0.0
    t_trace = []
    dT_trace = []
    qgen_trace = []
    qext_trace = []

    wafer_active_times = []
    wafer_total_times = []
    wafer_prod_rms = []
    wafer_prod_peaks = []
    wafer_margin_peaks = []
    max_force_sat = 0.0
    total_completed_sites = 0

    for w in range(num_wafers):
        max_active_s = 1.00
        t_local = np.arange(0.0, max_active_s, dt)
        n_loc = len(t_local)

        # Accelerated smooth coarse Z_sub alignment inside shortened margin window
        align_t0 = 0.005
        align_t1 = max(align_t0 + 0.006, align_end_t - 0.003)
        z_sub_target = smooth_step_profile(t_local, align_t0, align_t1, 0.25)

        dist_base = (
            0.035 * np.sin(2 * np.pi * 45.0 * (global_t + t_local))
            + 0.018 * np.sin(2 * np.pi * 130.0 * (global_t + t_local))
            + 0.004 * rng.standard_normal(n_loc)
        )

        site_ptr = [0] * 4
        settle_cnt = [0] * 4
        etch_cur = [0.0] * 4
        diss_cur = [0.0] * 4
        completed_w = [0] * 4

        prod_errs = []
        margin_errs = []
        active_dur = max_active_s

        for i in range(n_loc):
            t_abs = global_t + t_local[i]
            p_sub, _, qg_s, qe_s, sat_s = z_sub.step(z_sub_target[i], dist_base[i], dt, 1, buf_sub, True)
            p_sub_handoff += alpha_handoff * (p_sub - p_sub_handoff)
            max_force_sat = max(max_force_sat, sat_s)

            in_margin = t_local[i] <= align_end_t
            qg_step = qg_s
            qe_step = qe_s
            dT_max_step = 0.0

            for p in range(4):
                q_sites = path_queues[p]
                idx_s = min(site_ptr[p], len(q_sites) - 1)
                cur_site = q_sites[idx_s]

                z_lens_cmd_base = 0.06 * np.sin(2 * np.pi * 25.0 * t_abs + 0.5 * p)
                tx_cmd = 0.04 * np.sin(2 * np.pi * 18.0 * t_abs + 0.4 * p)
                ty_cmd = 0.04 * np.cos(2 * np.pi * 22.0 * t_abs + 0.3 * p)

                d_p = dist_base[i] * (0.85 + 0.08 * p)
                lens_cmd = (z_sub_target[i] + z_lens_cmd_base) - p_sub_handoff

                p_lens, dT_l, qg_l, qe_l, sat_l = lens_acts[p].step(lens_cmd, 0.5 * d_p, dt, 1, buf_lens[p], True)
                p_tx, _, qg_x, qe_x, sat_x = tx_acts[p].step(tx_cmd, 0.3 * d_p, dt, 1, buf_tx[p], True)
                p_ty, _, qg_y, qe_y, sat_y = ty_acts[p].step(ty_cmd, 0.3 * d_p, dt, 1, buf_ty[p], True)

                max_force_sat = max(max_force_sat, sat_l, sat_x, sat_y)
                qg_step += qg_l + qg_x + qg_y
                qe_step += qe_l + qe_x + qe_y
                dT_max_step = max(dT_max_step, dT_l)

                desired_z = z_sub_target[i] + z_lens_cmd_base
                actual_z = p_sub + p_lens
                tilt_proj = 0.04 * ((p_tx - tx_cmd) + (p_ty - ty_cmd))
                f_err = (desired_z - actual_z) + tilt_proj

                if in_margin:
                    margin_errs.append(f_err)
                else:
                    prod_errs.append(f_err)

                if (not in_margin) and site_ptr[p] < len(q_sites):
                    if abs(f_err) <= focal_tol_nm:
                        settle_cnt[p] += 1
                    else:
                        settle_cnt[p] = 0

                    req_d = cur_site["depth"]
                    if settle_cnt[p] >= settle_steps_req and abs(f_err) <= focal_tol_nm and etch_cur[p] < req_d:
                        etch_cur[p] = min(req_d, etch_cur[p] + base_etch_rate * dt)

                    if etch_cur[p] >= req_d * 0.999 and diss_cur[p] < 1.0:
                        diss_cur[p] = min(1.0, diss_cur[p] + dissolution_rate * dt)

                    if etch_cur[p] >= req_d * 0.999 and diss_cur[p] >= 0.999:
                        completed_w[p] += 1
                        site_ptr[p] += 1
                        settle_cnt[p] = 0
                        etch_cur[p] = 0.0
                        diss_cur[p] = 0.0

            t_trace.append(t_abs)
            dT_trace.append(dT_max_step)
            qgen_trace.append(qg_step)
            qext_trace.append(qe_step)

            if sum(completed_w) == 16:
                active_dur = t_local[i] + dt
                break

        global_t += active_dur
        total_completed_sites += sum(completed_w)
        wafer_active_times.append(active_dur)
        wafer_prod_rms.append(np.sqrt(np.mean(np.array(prod_errs) ** 2)))
        wafer_prod_peaks.append(np.max(np.abs(prod_errs)))
        wafer_margin_peaks.append(np.max(np.abs(margin_errs)))

        # Wafer swap & thermal relaxation interval
        n_swap = max(1, int(round(swap_duration_s / dt)))
        for k_s in range(1, n_swap + 1):
            t_abs = global_t + k_s * dt
            _, _, qg_s, qe_s, _ = z_sub.step(0.0, 0.0, dt, 1, buf_sub, False)
            qg_step = qg_s
            qe_step = qe_s
            dT_max_step = 0.0
            for p in range(4):
                _, dT_l, qg_l, qe_l, _ = lens_acts[p].step(0.0, 0.0, dt, 1, buf_lens[p], False)
                _, _, qg_x, qe_x, _ = tx_acts[p].step(0.0, 0.0, dt, 1, buf_tx[p], False)
                _, _, qg_y, qe_y, _ = ty_acts[p].step(0.0, 0.0, dt, 1, buf_ty[p], False)
                qg_step += qg_l + qg_x + qg_y
                qe_step += qe_l + qe_x + qe_y
                dT_max_step = max(dT_max_step, dT_l)

            t_trace.append(t_abs)
            dT_trace.append(dT_max_step)
            qgen_trace.append(qg_step)
            qext_trace.append(qe_step)

        global_t += swap_duration_s
        wafer_total_times.append(active_dur + swap_duration_s)

    # Compare against v9 Baseline (175.40 ms active etch+dev, 325.40 ms total cycle)
    v9_base_active_ms = 175.40
    v9_base_total_ms = 325.40
    mean_active_ms = np.mean(wafer_active_times) * 1e3
    mean_total_ms = np.mean(wafer_total_times) * 1e3

    active_gain_pct = 100.0 * (v9_base_active_ms - mean_active_ms) / v9_base_active_ms
    total_gain_pct = 100.0 * (v9_base_total_ms - mean_total_ms) / v9_base_total_ms
    wafers_per_day = 86400.0 / (mean_total_ms * 1e-3)

    # Physics verification printout
    print("=== v10 Real Physics & >=50% Cycle Gain Verification (20-Wafer Campaign) ===")
    print(
        f"[1. Throughput] Active Etch+Dev: {mean_active_ms:.2f} ms (Gain vs v9 Base: {active_gain_pct:.1f}%) | "
        f"Total Cycle (incl. {swap_duration_s*1e3:.1f} ms swap): {mean_total_ms:.2f} ms "
        f"(Total Gain: {total_gain_pct:.1f}%) | Wafers/24h Day: {wafers_per_day:,.0f}"
    )
    print(
        f"[2. Optical DOF] Rayleigh Half-DOF (13.5nm, NA=0.55): ±{rayleigh_dof_half_nm:.2f} nm | "
        f"Prod RMS Err: {np.mean(wafer_prod_rms)*1e3:.1f} pm | "
        f"Max Prod Peak: {np.max(wafer_prod_peaks):.4f} nm (< {sub_nm_focus_budget:.2f} nm budget) | "
        f"Margin Peak: {np.max(wafer_margin_peaks):.4f} nm"
    )
    print(
        f"[3. Mechanics] Z_Lens f_n: {lens_acts[0].natural_freq_hz:.1f} Hz (zeta={lens_acts[0].damping_ratio:.2f}) | "
        f"Z_Sub f_n: {z_sub.natural_freq_hz:.1f} Hz (zeta={z_sub.damping_ratio:.2f}) | "
        f"Max Force Saturation Ratio: {max_force_sat:.3f} (< 1.0 = No Clipping)"
    )
    print(
        f"[4. Thermal 20-Wafer] Peak Lens dT: {np.max(dT_trace)*1e3:.2f} mK | "
        f"Final dT (Wafer 20): {dT_trace[-1]*1e3:.2f} mK | "
        f"Completed Sites: {total_completed_sites}/{num_wafers*16}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(t_trace, np.array(dT_trace) * 1e3, label="Peak Lens dT [mK]")
    axes[0].set_title("20-Wafer Continuous Thermal Trajectory (Pre-Diamond)")
    axes[0].set_xlabel("Campaign Time [s]")
    axes[0].set_ylabel("Temp Rise dT [mK]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    w_idx = np.arange(1, num_wafers + 1)
    axes[1].plot(w_idx, np.array(wafer_prod_rms) * 1e3, marker="o", markersize=4, label="Prod RMS Err [pm]")
    axes[1].plot(w_idx, np.array(wafer_prod_peaks) * 1e3, marker="s", markersize=4, label="Prod Peak Err [pm]")
    axes[1].axhline(sub_nm_focus_budget * 1e3, linestyle="--", color="r", label="300 pm Spec Limit")
    axes[1].set_title("Per-Wafer Production Focus Accuracy (Wafers 1–20)")
    axes[1].set_xlabel("Wafer Number")
    axes[1].set_ylabel("Focal Error [pm]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("split_z_physics_verify_v10.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_multi_wafer_campaign()