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
    """Precision channel with hysteresis, hybrid SMC, and pre-diamond thermal-drain model."""

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

        f_ctrl = np.clip(
            self.k * delayed_cmd + self.kp * err + self.kd * d_err + self.smc_gain * sat,
            -self.max_force,
            self.max_force,
        )

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

        # During wafer swap/rest, active etch drive drops; passive thermal drain relaxes delta_T
        effort_ratio = f_ctrl / max(self.max_force, 1e-9)
        duty_factor = 1.0 if active_etching else 0.15
        q_gen = self.q_gen_scale * (effort_ratio ** 2) * duty_factor
        q_ext = self.delta_T / max(self.r_th, 1e-6)
        self.delta_T = max(0.0, self.delta_T + ((q_gen - q_ext) / max(self.c_th, 1e-6)) * dt)

        thermal_offset = self.thermal_drift * self.delta_T
        effective_pos = self.pos + thermal_offset
        return effective_pos, self.delta_T, q_gen, q_ext


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


def simulate_single_wafer_cycle(
    etch_rate_scale=1.0,
    swap_duration_s=0.15,
    core_power_ratio=4.0,
    optical_eta=0.85,
    handoff_tau=0.004,
):
    """
    Simulates 1 wafer active etch+development (margin alignment + 4-path disjoint quadrant etch)
    followed by a wafer swap/thermal-relaxation interval.
    Returns active etch+dev completion time [s], total cycle time including swap [s],
    peak delta_T [K], delta_T after swap [K], and RMS focal error [nm].
    """
    dt = 1e-4
    max_active_s = 1.50
    t_active = np.arange(0.0, max_active_s, dt)
    n_act = len(t_active)

    r_wafer = 2.00
    r_active = 1.65
    align_end_t = 0.055

    quadrants = {
        0: (-r_active, 0.0, -r_active, 0.0),
        1: (0.0, r_active, -r_active, 0.0),
        2: (-r_active, 0.0, 0.0, r_active),
        3: (0.0, r_active, 0.0, r_active),
    }

    base_depths = [8.0, 1.2, 6.0, 2.4]
    path_queues = []
    for p_idx in range(4):
        raw_p = [{"depth": base_depths[k] + 0.3 * p_idx} for k in range(4)]
        path_queues.append(order_alternating(raw_p))

    z_sub_target = smooth_step_profile(t_active, 0.020, 0.050, 0.25)
    z_sub = ActuatorChannel("Z_Sub", 1.0, 1600.0, 85.0, 80.0, 4.0, 1.5, 0.020, 0.008, kp=3800.0, kd=65.0)
    buf_sub = []

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
    buf_lens = [[] for _ in range(4)]
    buf_tx = [[] for _ in range(4)]
    buf_ty = [[] for _ in range(4)]

    rng = np.random.default_rng(53)
    dist_base = (
        0.035 * np.sin(2 * np.pi * 45.0 * t_active)
        + 0.018 * np.sin(2 * np.pi * 130.0 * t_active)
        + 0.004 * rng.standard_normal(n_act)
    )

    per_path_power_scale = (core_power_ratio / 4.0) * optical_eta
    base_etch_rate = 240.0 * per_path_power_scale * etch_rate_scale
    dissolution_rate = 360.0 * etch_rate_scale
    focal_tol_nm = 0.30
    settle_steps_req = 15

    site_ptr = [0] * 4
    settle_cnt = [0] * 4
    etch_cur = [0.0] * 4
    diss_cur = [0.0] * 4
    completed_per_path = [0] * 4

    p_sub_handoff = 0.0
    alpha_handoff = dt / max(handoff_tau + dt, 1e-9)

    prod_errs = []
    dT_hist = []
    qgen_hist = []
    qext_hist = []
    t_hist = []

    etch_dev_done_time = max_active_s

    for i in range(n_act):
        p_sub, _, qg_s, qe_s = z_sub.step(z_sub_target[i], dist_base[i], dt, 1, buf_sub, True)
        p_sub_handoff += alpha_handoff * (p_sub - p_sub_handoff)
        in_margin = t_active[i] <= align_end_t

        qg_step = qg_s
        qe_step = qe_s
        dT_step_max = 0.0

        for p in range(4):
            q_sites = path_queues[p]
            idx_s = min(site_ptr[p], len(q_sites) - 1)
            cur_site = q_sites[idx_s]

            z_lens_cmd_base = 0.06 * np.sin(2 * np.pi * 25.0 * t_active[i] + 0.5 * p)
            tx_cmd = 0.04 * np.sin(2 * np.pi * 18.0 * t_active[i] + 0.4 * p)
            ty_cmd = 0.04 * np.cos(2 * np.pi * 22.0 * t_active[i] + 0.3 * p)

            d_p = dist_base[i] * (0.85 + 0.08 * p)
            lens_cmd = (z_sub_target[i] + z_lens_cmd_base) - p_sub_handoff

            p_lens, dT_l, qg_l, qe_l = lens_acts[p].step(lens_cmd, 0.5 * d_p, dt, 1, buf_lens[p], True)
            p_tx, _, qg_x, qe_x = tx_acts[p].step(tx_cmd, 0.3 * d_p, dt, 1, buf_tx[p], True)
            p_ty, _, qg_y, qe_y = ty_acts[p].step(ty_cmd, 0.3 * d_p, dt, 1, buf_ty[p], True)

            qg_step += qg_l + qg_x + qg_y
            qe_step += qe_l + qe_x + qe_y
            dT_step_max = max(dT_step_max, dT_l)

            desired_z = z_sub_target[i] + z_lens_cmd_base
            actual_z = p_sub + p_lens
            tilt_proj = 0.04 * ((p_tx - tx_cmd) + (p_ty - ty_cmd))
            f_err = (desired_z - actual_z) + tilt_proj

            if not in_margin:
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
                    completed_per_path[p] += 1
                    site_ptr[p] += 1
                    settle_cnt[p] = 0
                    etch_cur[p] = 0.0
                    diss_cur[p] = 0.0

        t_hist.append(t_active[i])
        dT_hist.append(dT_step_max)
        qgen_hist.append(qg_step)
        qext_hist.append(qe_step)

        if sum(completed_per_path) == 16:
            etch_dev_done_time = t_active[i] + dt
            break

    dT_at_swap_start = dT_hist[-1]

    # Wafer exchange / rest window: thermal relaxation ("thermal dissolution") of delta_T
    n_swap = max(1, int(round(swap_duration_s / dt)))
    t_swap_arr = np.arange(1, n_swap + 1) * dt
    for k_step in range(n_swap):
        t_cur = etch_dev_done_time + t_swap_arr[k_step]
        qg_step = 0.0
        qe_step = 0.0
        dT_step_max = 0.0
        for p in range(4):
            _, dT_l, qg_l, qe_l = lens_acts[p].step(0.0, 0.0, dt, 1, buf_lens[p], active_etching=False)
            _, _, qg_x, qe_x = tx_acts[p].step(0.0, 0.0, dt, 1, buf_tx[p], active_etching=False)
            _, _, qg_y, qe_y = ty_acts[p].step(0.0, 0.0, dt, 1, buf_ty[p], active_etching=False)
            qg_step += qg_l + qg_x + qg_y
            qe_step += qe_l + qe_x + qe_y
            dT_step_max = max(dT_step_max, dT_l)

        t_hist.append(t_cur)
        dT_hist.append(dT_step_max)
        qgen_hist.append(qg_step)
        qext_hist.append(qe_step)

    dT_after_swap = dT_hist[-1]
    total_cycle_s = etch_dev_done_time + swap_duration_s
    prod_rms_nm = np.sqrt(np.mean(np.array(prod_errs) ** 2)) if prod_errs else 0.0

    return {
        "etch_dev_s": etch_dev_done_time,
        "swap_s": swap_duration_s,
        "total_cycle_s": total_cycle_s,
        "peak_dT_K": np.max(dT_hist),
        "dT_before_swap_K": dT_at_swap_start,
        "dT_after_swap_K": dT_after_swap,
        "prod_rms_nm": prod_rms_nm,
        "t_arr": np.array(t_hist),
        "dT_arr": np.array(dT_hist),
        "qgen_arr": np.array(qgen_hist),
        "qext_arr": np.array(qext_hist),
    }


def main():
    swap_s = 0.15  # 150 ms wafer feed/extract & thermal relaxation window per wafer

    # Baseline etch+development rate vs. ~1/4 reduced etch+development time (1 / 0.75 = 1.3333x rate)
    base_res = simulate_single_wafer_cycle(etch_rate_scale=1.0, swap_duration_s=swap_s)
    fast_res = simulate_single_wafer_cycle(etch_rate_scale=4.0 / 3.0, swap_duration_s=swap_s)

    t_base = base_res["etch_dev_s"]
    t_fast = fast_res["etch_dev_s"]
    abs_reduction_s = t_base - t_fast
    pct_reduction = 100.0 * abs_reduction_s / max(t_base, 1e-9)

    sec_per_day = 86400.0
    wafers_day_base = sec_per_day / base_res["total_cycle_s"]
    wafers_day_fast = sec_per_day / fast_res["total_cycle_s"]
    extra_wafers_day = wafers_day_fast - wafers_day_base

    thermal_relax_pct = (
        100.0 * (fast_res["dT_before_swap_K"] - fast_res["dT_after_swap_K"])
        / max(fast_res["dT_before_swap_K"], 1e-12)
    )

    print("=== v9 Multi-Wafer Etch/Development & Thermal Relaxation Summary ===")
    print(
        f"Baseline Etch+Dev Time: {t_base*1e3:.2f} ms | "
        f"Reduced Etch+Dev Time: {t_fast*1e3:.2f} ms | "
        f"Actual Decrease: {abs_reduction_s*1e3:.2f} ms ({pct_reduction:.2f}%)"
    )
    print(
        f"Wafer Swap/Relax Window: {swap_s*1e3:.1f} ms | "
        f"Total Cycle (Reduced): {fast_res['total_cycle_s']*1e3:.2f} ms | "
        f"Lens dT Relaxation During Swap: {fast_res['dT_before_swap_K']:.5f} K -> "
        f"{fast_res['dT_after_swap_K']:.5f} K ({thermal_relax_pct:.1f}% relaxed)"
    )
    print(
        f"24-Hour Production Throughput: {wafers_day_base:,.0f} wafers/day (Baseline) -> "
        f"{wafers_day_fast:,.0f} wafers/day (Reduced) | "
        f"Gain: +{extra_wafers_day:,.0f} wafers/day | "
        f"Prod RMS Err: {fast_res['prod_rms_nm']:.4f} nm"
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(fast_res["t_arr"] * 1e3, fast_res["dT_arr"] * 1e3, label="Peak Lens dT [mK]")
    axes[0].axvline(fast_res["etch_dev_s"] * 1e3, linestyle="--", color="r", label="Wafer Swap / Rest Start")
    axes[0].set_title("Wafer Etch + Swap Thermal Relaxation")
    axes[0].set_xlabel("Time [ms]")
    axes[0].set_ylabel("Temp Rise dT [mK]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    labels = ["Baseline\n(1.0x Rate)", "~1/4 Reduced\nEtch+Dev Time"]
    vals = [wafers_day_base, wafers_day_fast]
    axes[1].bar(labels, vals)
    axes[1].set_title("24-Hour Production Wafers / Day")
    axes[1].set_ylabel("Wafers / 24h Day")
    axes[1].grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("split_z_wafer_day_v9.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()