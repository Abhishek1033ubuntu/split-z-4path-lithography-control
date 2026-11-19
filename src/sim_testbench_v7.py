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

    def step(self, target_pos, disturbance_force, dt, latency_steps=1, cmd_history=None):
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

        effort_ratio = f_ctrl / max(self.max_force, 1e-9)
        q_gen = self.q_gen_scale * (effort_ratio ** 2)
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


def run_v7_four_path(core_power_ratio=4.0, optical_eta=0.85, handoff_tau=0.004):
    """
    v7: Coordinated split-Z handoff around early coarse alignment bump +
    extended 2.0 s window to inspect longer-run thermal behavior on pre-diamond baseline.
    """
    dt = 1e-4          # 0.1 ms step (10 kHz loop)
    duration = 2.00    # 2.0 s extended thermal window
    t = np.arange(0.0, duration, dt)
    n = len(t)

    quadrants = {
        0: (0.0, 2.0, 0.0, 2.0),  # Path 1: bottom-left
        1: (2.0, 4.0, 0.0, 2.0),  # Path 2: bottom-right
        2: (0.0, 2.0, 2.0, 4.0),  # Path 3: top-left
        3: (2.0, 4.0, 2.0, 4.0),  # Path 4: top-right
    }

    base_depths = [8.0, 1.2, 6.0, 2.4]
    path_queues = []
    for p_idx in range(4):
        xmin, xmax, ymin, ymax = quadrants[p_idx]
        x_centers = [xmin + 0.5, xmin + 1.5, xmin + 0.5, xmin + 1.5]
        y_centers = [ymin + 0.5, ymin + 0.5, ymin + 1.5, ymin + 1.5]
        raw_p = [
            {
                "depth": base_depths[k] + 0.3 * p_idx,
                "x": x_centers[k],
                "y": y_centers[k],
            }
            for k in range(4)
        ]
        path_queues.append(order_alternating(raw_p))

    z_sub_target = smooth_step_profile(t, 0.020, 0.050, 0.25)
    z_sub = ActuatorChannel(
        "Z_Sub", 1.0, 1600.0, 85.0, 80.0, 4.0, 1.5, 0.020, 0.008, kp=3800.0, kd=65.0
    )
    buf_sub = []

    # Pre-diamond thermal-drain baseline across per-path actuators
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

    rng = np.random.default_rng(37)
    dist_base = (
        0.035 * np.sin(2 * np.pi * 45.0 * t)
        + 0.018 * np.sin(2 * np.pi * 130.0 * t)
        + 0.004 * rng.standard_normal(n)
    )

    per_path_power_scale = (core_power_ratio / 4.0) * optical_eta
    base_etch_rate = 240.0 * per_path_power_scale
    dissolution_rate = 360.0
    focal_tol_nm = 0.30
    settle_steps_req = 15

    site_ptr = [0] * 4
    settle_cnt = [0] * 4
    etch_cur = [0.0] * 4
    diss_cur = [0.0] * 4
    completed_per_path = [0] * 4
    boundary_violations = 0

    # Smoothed substrate offset estimate for split-Z handoff to lens focus actuators
    p_sub_handoff = 0.0
    alpha_handoff = dt / max(handoff_tau + dt, 1e-9)

    focal_err_paths = np.zeros((4, n))
    q_gen_total = np.zeros(n)
    q_ext_total = np.zeros(n)
    max_dT_path = np.zeros(4)

    for i in range(n):
        p_sub, _, qg_s, qe_s = z_sub.step(z_sub_target[i], dist_base[i], dt, 1, buf_sub)
        p_sub_handoff += alpha_handoff * (p_sub - p_sub_handoff)

        qg_step = qg_s
        qe_step = qe_s

        for p in range(4):
            q_sites = path_queues[p]
            idx_s = min(site_ptr[p], len(q_sites) - 1)
            cur_site = q_sites[idx_s]

            xmin, xmax, ymin, ymax = quadrants[p]
            x_pos = np.clip(cur_site["x"], xmin + 1e-6, xmax - 1e-6)
            y_pos = np.clip(cur_site["y"], ymin + 1e-6, ymax - 1e-6)
            if not (xmin <= x_pos <= xmax and ymin <= y_pos <= ymax):
                boundary_violations += 1

            z_lens_cmd_base = 0.06 * np.sin(2 * np.pi * 25.0 * t[i] + 0.5 * p)
            tx_cmd = 0.04 * np.sin(2 * np.pi * 18.0 * t[i] + 0.4 * p)
            ty_cmd = 0.04 * np.cos(2 * np.pi * 22.0 * t[i] + 0.3 * p)

            d_p = dist_base[i] * (0.85 + 0.08 * p)
            # Coordinated split-Z handoff: use smoothed p_sub_handoff to avoid sharp transient kick
            lens_cmd = (z_sub_target[i] + z_lens_cmd_base) - p_sub_handoff

            p_lens, dT_l, qg_l, qe_l = lens_acts[p].step(lens_cmd, 0.5 * d_p, dt, 1, buf_lens[p])
            p_tx, _, qg_x, qe_x = tx_acts[p].step(tx_cmd, 0.3 * d_p, dt, 1, buf_tx[p])
            p_ty, _, qg_y, qe_y = ty_acts[p].step(ty_cmd, 0.3 * d_p, dt, 1, buf_ty[p])

            qg_step += qg_l + qg_x + qg_y
            qe_step += qe_l + qe_x + qe_y
            max_dT_path[p] = max(max_dT_path[p], dT_l)

            desired_z = z_sub_target[i] + z_lens_cmd_base
            actual_z = p_sub + p_lens
            tilt_proj = 0.04 * ((p_tx - tx_cmd) + (p_ty - ty_cmd))
            f_err = (desired_z - actual_z) + tilt_proj
            focal_err_paths[p, i] = f_err

            if site_ptr[p] < len(q_sites):
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

        q_gen_total[i] = qg_step
        q_ext_total[i] = qe_step

    grid_n = 64
    half = grid_n // 2
    final_map = np.zeros((grid_n, grid_n), dtype=int)
    quad_slices = {
        0: (slice(0, half), slice(0, half)),
        1: (slice(0, half), slice(half, grid_n)),
        2: (slice(half, grid_n), slice(0, half)),
        3: (slice(half, grid_n), slice(half, grid_n)),
    }
    for p, (sy, sx) in quad_slices.items():
        if completed_per_path[p] == len(path_queues[p]):
            final_map[sy, sx] += 1

    missed_cells = int(np.sum(final_map == 0))
    overlapped_cells = int(np.sum(final_map > 1))
    rms_all = np.sqrt(np.mean(focal_err_paths ** 2))
    peak_abs_err = np.max(np.abs(focal_err_paths))
    mean_err = np.mean(focal_err_paths)

    print(
        f"[v7 4-Path (2.0s)] RMS Err: {rms_all:.4f} nm ({rms_all*1e3:.1f} pm) | "
        f"Peak Abs Err: {peak_abs_err:.4f} nm | Mean Err: {mean_err:+.4f} nm | "
        f"Peak dT: {np.max(max_dT_path):.4f} K | "
        f"Final (Q_gen - Q_ext): {q_gen_total[-1] - q_ext_total[-1]:+.5f} W | "
        f"Completed: {sum(completed_per_path)}/16 | "
        f"Missed: {missed_cells} | Overlapped: {overlapped_cells} | "
        f"Violations: {boundary_violations}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    t_ms = t * 1e3

    for p in range(4):
        axes[0].plot(t_ms, focal_err_paths[p], label=f"Path {p+1}")
    axes[0].axhline(0.30, linestyle="--", linewidth=1, color="k", alpha=0.6)
    axes[0].axhline(-0.30, linestyle="--", linewidth=1, color="k", alpha=0.6)
    axes[0].axhline(0.0, linestyle=":", linewidth=1, color="gray", alpha=0.7)
    axes[0].set_xlim(0, 250)  # Zoom into early alignment & active etch window
    axes[0].set_title("v7 Per-Path Focal Error [nm] (0–250 ms)")
    axes[0].set_xlabel("Time [ms]")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, q_gen_total * 1e3, label="Total Q_gen [mW]")
    axes[1].plot(t, q_ext_total * 1e3, label="Total Q_ext [mW]")
    axes[1].set_title("v7 2.0s Thermal Power [mW]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    im = axes[2].imshow(final_map, origin="lower", cmap="Blues", vmin=0, vmax=2)
    axes[2].set_title("Disjoint 4-Quad Coverage (1=Clean, >1=Overlap)")
    axes[2].set_xlabel("X Grid")
    axes[2].set_ylabel("Y Grid")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig("split_z_4path_v7.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_v7_four_path(core_power_ratio=4.0, optical_eta=0.85, handoff_tau=0.004)