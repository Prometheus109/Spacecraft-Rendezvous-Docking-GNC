import numpy as np


class Controller:

    def __init__(self, cfg=None):
        self.u_max = 0.005
        self.dt = 15.0
        self.alpha_d = 0.25

        # Scenario dynamic limits
        self.speed_a = 0.020
        self.speed_b = 0.005
        self.corridor_w0 = 0.4
        self.corridor_k = 0.085
        self.terminal_vel = 0.018  # Safe docking target velocity (< 0.030 m/s)
        self.pred_steps = 4

        # Observer memory
        self.d_hat = np.zeros(2)
        self.prev_s = None
        self.prev_u = None
        self.Ad = None
        self.Bd = None

    def reset(self, scenario):
        """Called once before each flight run."""
        self.dt = float(scenario.get("dt", 15.0))
        self.u_max = float(scenario.get("u_max", 0.005))

        self.speed_a = float(scenario.get("speed_a", 0.020))
        self.speed_b = float(scenario.get("speed_b", 0.005))
        self.corridor_w0 = float(scenario.get("corridor_w0", 0.4))
        self.corridor_k = float(scenario.get("corridor_k", 0.085))
        self.terminal_vel = float(scenario.get("terminal_vel_mps", 0.018))

        pred_lead = float(scenario.get("prediction_lead_s", 60.0))
        self.pred_steps = max(1, int(round(pred_lead / self.dt)))

        self.d_hat = np.zeros(2)
        self.prev_s = None
        self.prev_u = None

        if "Ad" in scenario and "Bd" in scenario:
            self.Ad = np.array(scenario["Ad"], dtype=float)
            self.Bd = np.array(scenario["Bd"], dtype=float)
        else:
            n = float(scenario.get("n_orbit", 0.0011))
            tau = self.dt

            if abs(n) < 1e-8:
                self.Ad = np.array(
                    [
                        [1.0, 0.0, tau, 0.0],
                        [0.0, 1.0, 0.0, tau],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                    dtype=float,
                )
                self.Bd = np.array(
                    [
                        [0.5 * tau**2, 0.0],
                        [0.0, 0.5 * tau**2],
                        [tau, 0.0],
                        [0.0, tau],
                    ],
                    dtype=float,
                )
            else:
                nt = n * tau
                sin_nt = np.sin(nt)
                cos_nt = np.cos(nt)

                self.Ad = np.array(
                    [
                        [
                            4.0 - 3.0 * cos_nt,
                            0.0,
                            sin_nt / n,
                            2.0 * (1.0 - cos_nt) / n,
                        ],
                        [
                            6.0 * (sin_nt - nt),
                            1.0,
                            -2.0 * (1.0 - cos_nt) / n,
                            (4.0 * sin_nt - 3.0 * nt) / n,
                        ],
                        [3.0 * n * sin_nt, 0.0, cos_nt, 2.0 * sin_nt],
                        [
                            6.0 * n * (cos_nt - 1.0),
                            0.0,
                            -2.0 * sin_nt,
                            4.0 * cos_nt - 3.0,
                        ],
                    ],
                    dtype=float,
                )

                self.Bd = np.array(
                    [
                        [
                            (1.0 - cos_nt) / (n**2),
                            2.0 * (nt - sin_nt) / (n**2),
                        ],
                        [
                            -2.0 * (nt - sin_nt) / (n**2),
                            4.0 * (1.0 - cos_nt) / (n**2) - 1.5 * (tau**2),
                        ],
                        [sin_nt / n, 2.0 * (1.0 - cos_nt) / n],
                        [-2.0 * (1.0 - cos_nt) / n, (4.0 * sin_nt - 3.0 * nt) / n],
                    ],
                    dtype=float,
                )

    def _update_disturbance(self, s):
        """Filters velocity residual to isolate drag/acceleration bias."""
        if self.prev_s is not None and self.prev_u is not None:
            s_pred = self.Ad @ self.prev_s + self.Bd @ self.prev_u
            e_v = s[2:] - s_pred[2:]
            d_meas = e_v / self.dt
            self.d_hat = (1.0 - self.alpha_d) * self.d_hat + self.alpha_d * d_meas

    def step(self, t, s):
        """Predictive phase-plane control law."""
        s = np.array(s, dtype=float)
        x, y, vx, vy = s

        self._update_disturbance(s)

        # Acceleration budgets considering disturbance estimate
        a_net_x = max(1e-4, self.u_max - abs(self.d_hat[0]))
        a_net_y = max(1e-4, self.u_max - abs(self.d_hat[1]))

        # --- 1. Predictive Speed Limit (Prevents limit violations) ---
        margin_v = 0.005
        v_cap_future = (
            self.speed_a + self.speed_b * max(0.0, y) - margin_v
        ) / (1.0 + self.speed_b * self.dt)
        v_cap = max(0.010, v_cap_future)

        # --- 2. Y-Axis Guidance (Braking Parabola & Descent) ---
        y_dock = 0.5
        v_brake_y = np.sqrt(
            self.terminal_vel**2 + 2.0 * a_net_y * max(0.0, y - y_dock)
        )

        vy_max_allowed = min(v_cap, v_brake_y)
        if abs(y) > 0.4:
            # Maintain active approach speed (85% of budget, floor at 0.015 m/s)
            vy_desired = max(0.015, 0.85 * vy_max_allowed)
            vy_desired = min(vy_desired, vy_max_allowed)
        else:
            vy_desired = min(self.terminal_vel, vy_max_allowed)

        vy_target = -np.sign(y) * vy_desired if abs(y) > 0.02 else 0.0

        # --- 3. X-Axis Guidance (Corridor & Velocity Allocation) ---
        vx_budget = np.sqrt(max(0.0, v_cap**2 - vy_target**2))

        x_limit = self.corridor_w0 + self.corridor_k * max(0.0, y)
        dist_to_wall = max(0.0, x_limit - abs(x) - 0.12)
        vx_wall_max = np.sqrt(2.0 * a_net_x * dist_to_wall)

        vx_desired = min(0.08 * abs(x), vx_wall_max, vx_budget)
        vx_target = -np.sign(x) * vx_desired if abs(x) > 0.01 else 0.0

        # Calculate required accelerations
        ax_req = (vx_target - vx) / self.dt - self.d_hat[0]
        ay_req = (vy_target - vy) / self.dt - self.d_hat[1]

        # Clip thrust commands to max thruster limits
        ax = float(np.clip(ax_req, -self.u_max, self.u_max))
        ay = float(np.clip(ay_req, -self.u_max, self.u_max))
        u_cmd = np.array([ax, ay])

        # State prediction 60s ahead
        s_sim = s.copy()
        for _ in range(self.pred_steps):
            s_sim = self.Ad @ s_sim + self.Bd @ u_cmd + self.Bd @ self.d_hat

        pred_pos = [float(s_sim[0]), float(s_sim[1])]

        self.prev_s = s.copy()
        self.prev_u = u_cmd.copy()

        return {"u": [ax, ay], "pred_pos": pred_pos}