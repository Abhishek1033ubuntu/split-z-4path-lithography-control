import numpy as np
import matplotlib.pyplot as plt


class HysteresisInverter:
    """Rate-aware hysteresis pre-compensator for sub-nm positioning."""

    def __init__(self, hyst_amp=0.03, rate_coeff=0.0015):
        self.hyst_amp = hyst_amp
        self.rate_coeff = rate_coeff
        self.prev_target = 0.0

    def invert(self, desired_pos, dt):
        rate = (desired_pos - self.prev_target) / max(dt, 1e-9)
        eff_hyst = self.hyst_amp * (1.0 + self.rate_coeff * min(abs(rate), 120.0))
        u_cmd = desired_pos + eff_hyst * np.tanh(desired_pos) * np.tanh(rate * 1e-3)
        self.prev_target = desired_pos
        return u_cmd


class ActuatorChannel:
    """Precision channel with unclipped force authority, hysteresis, SMC, and thermal balance."""

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
        hyst_amp=0.03,
        kp=220.0,
        kd=14.0,
        smc_gain=1.2,
        boundary_layer=0.05,
        init_pos=0.0,
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

        self.pos = float(init_pos)
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

        f_raw = 0.25 * self.k * delayed_cmd + self.kp * err + self.kd * d_err + self.smc_gain * sat
        sat_ratio = abs(f_raw) / max(self.max_force, 1e-9)
        f_ctrl = np.clip(f_raw, -self.max_force, self.max_force)

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


def run_single_stress_case(seed, dist_scale=1.0, init_offset=0.0):
    dt = 1e-4
    num_wafers = 10
    align_end_t = 0.028
    settle_steps_req = 5
    etch_rate_scale = 2.35
    swap_duration_s = 0.075
    core_power_ratio = 4.0
    optical_eta = 0.85
    handoff_tau = 0.004
    focal_tol_nm = 0.30
    margin_guard_nm = 0.25

    z_sub = ActuatorChannel(
        "Z_Sub", 1.0, 1600.0, 85.0, 1500.0, 4.0, 1.5, 0.020, 0.008,
        kp=420.0, kd=45.0, smc_gain=1.5, init_pos=init_offset
    )
    lens_acts = [
        ActuatorChannel(
            f"Z_Lens_P{p+1}", 0.14, 2800.0, 55.0, 1500.0, 4.5, 1.2, 0.025, 0.010,
            kp=450.0, kd=22.0, smc_gain=1.2, init_pos=init_offset * 0.5
        )
        for p in range(4)
    ]
    tx_acts = [
        ActuatorChannel(
            f"TX_P{p+1}", 0.16, 2600.0, 50.0, 1500.0, 4.5, 1.2, 0.020, 0.008,
            kp=400.0, kd=20.0, smc_gain=1.0, init_pos=init_offset * 0.4
        )
        for p in range(4)
    ]
    ty_acts = [
        ActuatorChannel(
            f"TY_P{p+1}", 0.16, 2600.0, 50.0, 1500.0, 4.5, 1.2, 0.020, 0.008,
            kp=400.0, kd=20.0, smc_gain=1.0, init_pos=init_offset * 0.4
        )
        for p in range(4)
    ]

    buf_sub = []
    buf_lens = [[] for _ in range(4)]
    buf_tx = [[] for _ in range(4)]
    buf_ty = [[] for _ in range(4)]

    rng = np.random.default_rng(seed)
    base_etch_rate = 240.0 * (core_power_ratio / 4.0) * optical_eta * etch_rate_scale
    dissolution_rate = 360.0 * etch_rate_scale

    base_depths = [8.0, 1.2, 6.0, 2.4]
    path_queues = [
        order_alternating([{"depth": base_depths[k] + 0.3 * p_idx} for k in range(4)])
        for p_idx in range(4)
    ]

    p_sub_handoff = init_offset
    alpha_handoff = dt / max(handoff_tau + dt, 1e-9)

    global_t = 0.0
    all_prod_rms = []
    all_prod_peaks = []
    all_margin_peaks = []
    max_sat = 0.0
    max_dT_mk = 0.0
    completed_sites = 0

    for _ in range(num_wafers):
        t_local = np.arange(0.0, 1.00, dt)
        n_loc = len(t_local)
        z_sub_target = smooth_step_profile(t_local, 0.008, 0.022, 0.14)

        dist_base = dist_scale * (
            0.030 * np.sin(2 * np.pi * 45.0 * (global_t + t_local))
            + 0.015 * np.sin(2 * np.pi * 130.0 * (global_t + t_local))
            + 0.003 * rng.standard_normal(n_loc)
        )

        site_ptr = [0] * 4
        settle_cnt = [0] * 4
        etch_cur = [0.0] * 4
        diss_cur = [0.0] * 4
        completed_w = [0] * 4
        prod_errs = []
        margin_errs = []
        active_dur = 1.00

        for i in range(n_loc):
            t_abs = global_t + t_local[i]
            p_sub, _, _, _, sat_s = z_sub.step(z_sub_target[i], dist_base[i], dt, 1, buf_sub, True)
            p_sub_handoff += alpha_handoff * (p_sub - p_sub_handoff)
            max_sat = max(max_sat, sat_s)

            in_margin = t_local[i] <= align_end_t

            for p in range(4):
                q_sites = path_queues[p]
                cur_site = q_sites[min(site_ptr[p], len(q_sites) - 1)]

                z_lens_cmd_base = 0.05 * np.sin(2 * np.pi * 25.0 * t_abs + 0.5 * p)
                tx_cmd = 0.035 * np.sin(2 * np.pi * 18.0 * t_abs + 0.4 * p)
                ty_cmd = 0.035 * np.cos(2 * np.pi * 22.0 * t_abs + 0.3 * p)

                d_p = dist_base[i] * (0.85 + 0.08 * p)
                lens_cmd = (z_sub_target[i] + z_lens_cmd_base) - p_sub_handoff

                p_lens, dT_l, _, _, sat_l = lens_acts[p].step(lens_cmd, 0.5 * d_p, dt, 1, buf_lens[p], True)
                p_tx, _, _, _, sat_x = tx_acts[p].step(tx_cmd, 0.3 * d_p, dt, 1, buf_tx[p], True)
                p_ty, _, _, _, sat_y = ty_acts[p].step(ty_cmd, 0.3 * d_p, dt, 1, buf_ty[p], True)

                max_sat = max(max_sat, sat_l, sat_x, sat_y)
                max_dT_mk = max(max_dT_mk, dT_l * 1e3)

                desired_z = z_sub_target[i] + z_lens_cmd_base
                actual_z = p_sub + p_lens
                tilt_proj = 0.04 * ((p_tx - tx_cmd) + (p_ty - ty_cmd))
                f_err = (desired_z - actual_z) + tilt_proj

                if in_margin:
                    margin_errs.append(f_err)
                else:
                    prod_errs.append(f_err)

                if (not in_margin) and abs(f_err) <= margin_guard_nm and site_ptr[p] < len(q_sites):
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

            if sum(completed_w) == 16:
                active_dur = t_local[i] + dt
                break

        global_t += active_dur
        completed_sites += sum(completed_w)
        all_prod_rms.append(np.sqrt(np.mean(np.array(prod_errs) ** 2)))
        all_prod_peaks.append(np.max(np.abs(prod_errs)))
        all_margin_peaks.append(np.max(np.abs(margin_errs)))

        n_swap = max(1, int(round(swap_duration_s / dt)))
        for _ in range(n_swap):
            z_sub.step(0.0, 0.0, dt, 1, buf_sub, False)
            for p in range(4):
                _, dT_l, _, _, _ = lens_acts[p].step(0.0, 0.0, dt, 1, buf_lens[p], False)
                tx_acts[p].step(0.0, 0.0, dt, 1, buf_tx[p], False)
                ty_acts[p].step(0.0, 0.0, dt, 1, buf_ty[p], False)
                max_dT_mk = max(max_dT_mk, dT_l * 1e3)
        global_t += swap_duration_s

    return {
        "prod_rms_pm": float(np.mean(all_prod_rms) * 1e3),
        "prod_peak_nm": float(np.max(all_prod_peaks)),
        "margin_peak_nm": float(np.max(all_margin_peaks)),
        "max_sat": float(max_sat),
        "max_dT_mk": float(max_dT_mk),
        "completed": int(completed_sites),
        "target_sites": int(num_wafers * 16),
    }


def main():
    seeds = [11, 23, 37, 53, 83]
    scenarios = [
        ("Nominal_1x_Dist", 1.0, 0.0),
        ("Stress_2.5x_Dist+Offset", 2.5, 0.08),
    ]

    summary_rows = []
    for label, d_scale, offset in scenarios:
        res_list = [run_single_stress_case(s, d_scale, offset) for s in seeds]
        worst_margin = max(r["margin_peak_nm"] for r in res_list)
        worst_prod_peak = max(r["prod_peak_nm"] for r in res_list)
        mean_rms_pm = np.mean([r["prod_rms_pm"] for r in res_list])
        worst_sat = max(r["max_sat"] for r in res_list)
        worst_dT_mk = max(r["max_dT_mk"] for r in res_list)
        tot_done = sum(r["completed"] for r in res_list)
        tot_req = sum(r["target_sites"] for r in res_list)

        print(
            f"[{label}] Worst Margin Peak: {worst_margin:.4f} nm | "
            f"Worst Prod Peak: {worst_prod_peak:.4f} nm | "
            f"Mean Prod RMS: {mean_rms_pm:.1f} pm | "
            f"Max Sat: {worst_sat:.3f} | Peak dT: {worst_dT_mk:.4f} mK | "
            f"Sites: {tot_done}/{tot_req}"
        )
        summary_rows.append((label, worst_margin, worst_prod_peak))

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [row[0] for row in summary_rows]
    margin_vals = [row[1] for row in summary_rows]
    prod_vals = [row[2] for row in summary_rows]
    x = np.arange(len(labels))
    w = 0.32

    ax.bar(x - w / 2, margin_vals, width=w, label="Worst Margin Peak [nm]")
    ax.bar(x + w / 2, prod_vals, width=w, label="Worst Prod Peak [nm]")
    ax.axhline(0.25, linestyle="--", color="orange", label="250 pm Margin Guard")
    ax.axhline(0.30, linestyle="--", color="r", label="300 pm Spec Limit")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Peak Focal Error [nm]")
    ax.set_title("v12 Multi-Seed Stress Check (5 Seeds x 10 Wafers)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("split_z_stress_v12.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()